from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.models.olmo2 import (
    Olmo2DecoderLayer,
    dense_variant_forward_flops,
    dense_variant_parameter_breakdown,
)
from lm_from_zero.training import (
    BatchCursor,
    CausalBatchConfig,
    DenseTrainingConfig,
    OptimizationConfig,
    build_hybrid_muon,
    create_checkpoint_binding,
    partition_hybrid_muon,
    restore_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)


def _config() -> Olmo2Config:
    return Olmo2Config(
        model_name="variant-test",
        tokenizer_hash="0" * 64,
        vocab_size=272,
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
    )


class DenseVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1337)

    def test_all_variants_have_finite_forward_and_backward(self) -> None:
        input_ids = torch.randint(8, 272, (2, 7))
        for variant in (
            "baseline",
            "learned_absolute_positions",
            "layer_norm",
            "gelu",
            "mha",
            "without_qk_norm",
            "tied_embeddings",
        ):
            model = Olmo2ForCausalLM(_config(), variant=variant)
            output = model(input_ids, labels=input_ids)
            loss = cast(Tensor, output.loss)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()  # type: ignore[no-untyped-call]
            self.assertTrue(
                all(
                    bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )

    def test_position_and_attention_variants_preserve_cache_parity(self) -> None:
        input_ids = torch.randint(8, 272, (2, 8))
        for variant in ("learned_absolute_positions", "mha"):
            model = Olmo2ForCausalLM(_config(), variant=variant).eval()
            with torch.no_grad():
                full = model(input_ids).logits
                prefix = model(input_ids[:, :4], use_cache=True)
                suffix = model(
                    input_ids[:, 4:],
                    cache=prefix.cache,
                    use_cache=True,
                ).logits
            torch.testing.assert_close(suffix, full[:, 4:], atol=1e-5, rtol=1e-5)

    def test_variant_structure_and_tied_weight_contracts(self) -> None:
        config = _config()
        layer_norm = Olmo2ForCausalLM(config, variant="layer_norm")
        layer_norm_layer = cast(Olmo2DecoderLayer, layer_norm.layers[0])
        self.assertTrue(
            all(
                isinstance(module, nn.LayerNorm)
                for layer in (layer_norm_layer,)
                for module in (
                    layer.self_attn.q_norm,
                    layer.self_attn.k_norm,
                    layer.post_attention_layernorm,
                    layer.post_feedforward_layernorm,
                )
            )
        )
        gelu = Olmo2ForCausalLM(config, variant="gelu")
        gelu_layer = cast(Olmo2DecoderLayer, gelu.layers[0])
        gelu_up = gelu_layer.mlp.up_proj
        self.assertEqual(gelu_up.out_features, 96)
        self.assertFalse(hasattr(gelu_layer.mlp, "gate_proj"))
        without_qk = Olmo2ForCausalLM(config, variant="without_qk_norm")
        without_qk_layer = cast(Olmo2DecoderLayer, without_qk.layers[0])
        self.assertIsInstance(without_qk_layer.self_attn.q_norm, nn.Identity)
        self.assertIsInstance(without_qk_layer.self_attn.k_norm, nn.Identity)
        tied = Olmo2ForCausalLM(config, variant="tied_embeddings")
        self.assertEqual(
            tied.embed_tokens.weight.data_ptr(), tied.lm_head.weight.data_ptr()
        )

    def test_baseline_parameter_count_and_config_remain_canonical(self) -> None:
        config = _config()
        model = Olmo2ForCausalLM(config)
        self.assertEqual(model.trainable_parameter_count(), 36_096)
        self.assertEqual(model.config, config)
        self.assertEqual(model.variant, "baseline")

    def test_realized_variant_parameter_and_flop_accounting_matches_models(
        self,
    ) -> None:
        config = _config()
        for variant in (
            "baseline",
            "learned_absolute_positions",
            "layer_norm",
            "gelu",
            "mha",
            "without_qk_norm",
            "tied_embeddings",
        ):
            model = Olmo2ForCausalLM(config, variant=variant)
            self.assertEqual(
                dense_variant_parameter_breakdown(config, variant).total,
                model.trainable_parameter_count(),
            )
            flops = dense_variant_forward_flops(config, 8, variant)
            self.assertEqual(
                flops.projection_flops_per_token + flops.attention_flops_per_token,
                flops.total_flops_per_token,
            )

    def test_variant_controls_are_hashed_without_changing_baseline_hash(self) -> None:
        config = _config()
        batch = CausalBatchConfig(
            sequence_length=8,
            micro_batch_size=1,
            seed=1337,
        )
        optimization = OptimizationConfig(total_steps=4)
        baseline = DenseTrainingConfig(
            model=config,
            batch=batch,
            optimization=optimization,
        )
        explicit_baseline = baseline.model_copy(
            update={"model_variant": "baseline", "optimizer_variant": "adamw"}
        )
        gelu = baseline.model_copy(update={"model_variant": "gelu"})
        muon = baseline.model_copy(update={"optimizer_variant": "hybrid_muon"})
        self.assertEqual(baseline.config_hash, explicit_baseline.config_hash)
        self.assertNotEqual(baseline.config_hash, gelu.config_hash)
        self.assertNotEqual(baseline.config_hash, muon.config_hash)
        self.assertIn('"model_variant":"gelu"', gelu.canonical_json())
        self.assertIn('"optimizer_variant":"hybrid_muon"', muon.canonical_json())


class HybridMuonTests(unittest.TestCase):
    def test_partition_is_complete_and_optimizer_state_roundtrips(self) -> None:
        model = Olmo2ForCausalLM(_config())
        partition = partition_hybrid_muon(model)
        self.assertTrue(partition.muon_names)
        self.assertEqual(
            {id(parameter) for parameter in partition.all_parameters},
            {id(parameter) for parameter in model.parameters()},
        )
        optimizer, _ = build_hybrid_muon(
            model,
            OptimizationConfig(total_steps=4, gradient_clip_norm=1_000),
        )
        input_ids = torch.randint(8, 272, (1, 5))
        loss = cast(Tensor, model(input_ids, labels=input_ids).loss)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        state = optimizer.state_dict()
        restored, _ = build_hybrid_muon(
            Olmo2ForCausalLM(_config()),
            OptimizationConfig(total_steps=4, gradient_clip_norm=1_000),
        )
        restored.load_state_dict(state)
        self.assertEqual(set(state), {"state", "param_groups"})
        self.assertEqual(
            [group["group_name"] for group in state["param_groups"]],
            ["muon", "adamw_decay", "adamw_no_decay"],
        )

    def test_checkpoint_validator_accepts_hybrid_state(self) -> None:
        model = Olmo2ForCausalLM(_config())
        optimizer, _ = build_hybrid_muon(
            model,
            OptimizationConfig(total_steps=4, gradient_clip_norm=1_000),
        )
        input_ids = torch.randint(8, 272, (1, 5))
        loss = cast(Tensor, model(input_ids, labels=input_ids).loss)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        binding = create_checkpoint_binding(
            architecture="olmo2",
            resolved_model_config=model.config.model_dump(mode="json"),
            tokenizer_sha256=model.config.tokenizer_hash,
            shard_manifest_sha256="1" * 64,
            rank=0,
            world_size=1,
            repository=Path.cwd(),
        )
        cursor = BatchCursor(
            build_manifest_sha256="1" * 64,
            tokenizer_hash=model.config.tokenizer_hash,
            split="train",
            sequence_length=5,
            seed=1337,
            rank=0,
            world_size=1,
            shuffle=True,
            next_local_window=1,
            sequences_consumed=1,
            tokens_consumed=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(
                directory,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                binding=binding,
                optimizer_step=1,
                scheduler_step=1,
            )
            validate_checkpoint(checkpoint)
            restored_model = Olmo2ForCausalLM(_config())
            restored_optimizer, _ = build_hybrid_muon(
                restored_model,
                OptimizationConfig(total_steps=4, gradient_clip_norm=1_000),
            )
            restored = restore_checkpoint(
                checkpoint,
                model=restored_model,
                optimizer=restored_optimizer,
                expected_binding=binding,
            )
            self.assertEqual(restored.manifest.progress.optimizer_step, 1)


if __name__ == "__main__":
    unittest.main()
