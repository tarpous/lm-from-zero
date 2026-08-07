from __future__ import annotations

import unittest
from typing import cast

import torch
from torch import Tensor, nn

from lm_from_zero.models.config import Olmo2Config
from lm_from_zero.models.interfaces import DenseKVCache
from lm_from_zero.models.olmo2 import (
    GELUMLP,
    DenseModelVariant,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    RMSNorm,
    cache_sequence_length,
)


def tiny_config() -> Olmo2Config:
    return Olmo2Config.model_validate(
        {
            "model_name": "test-olmo2-variants",
            "tokenizer_hash": "0" * 64,
            "vocab_size": 272,
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "max_position_embeddings": 16,
        }
    )


class Olmo2VariantTests(unittest.TestCase):
    def test_variants_run_forward_backward_and_cache(self) -> None:
        config = tiny_config()
        input_ids = torch.randint(8, config.vocab_size, (2, 6))
        for variant in (
            "learned_absolute_positions",
            "layer_norm",
            "gelu",
            "mha",
            "without_qk_norm",
            "tied_embeddings",
        ):
            with self.subTest(variant=variant):
                torch.manual_seed(123)
                model = Olmo2ForCausalLM(config, variant=variant).eval()
                output = model(input_ids, labels=input_ids)
                loss = cast(Tensor, output.loss)
                self.assertTrue(bool(torch.isfinite(loss)))
                model.zero_grad(set_to_none=True)
                torch.autograd.backward(loss)
                self.assertTrue(
                    all(
                        parameter.grad is None
                        or bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                    )
                )

                with torch.no_grad():
                    expected = model(input_ids).logits
                    cache: DenseKVCache | None = None
                    pieces: list[Tensor] = []
                    for index in range(input_ids.shape[1]):
                        step = model(
                            input_ids[:, index : index + 1],
                            cache=cache,
                            use_cache=True,
                        )
                        cache = step.cache
                        pieces.append(step.logits)
                    actual = torch.cat(pieces, dim=1)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                self.assertIsNotNone(cache)
                self.assertEqual(
                    cache_sequence_length(cast(DenseKVCache, cache)),
                    input_ids.shape[1],
                )

    def test_variant_module_contracts_and_tied_weights(self) -> None:
        config = tiny_config()
        layer_norm_model = Olmo2ForCausalLM(config, variant="layer_norm")
        layer = cast(Olmo2DecoderLayer, layer_norm_model.layers[0])
        self.assertIsInstance(layer.self_attn.q_norm, nn.LayerNorm)
        self.assertIsInstance(layer.self_attn.k_norm, nn.LayerNorm)
        self.assertIsInstance(layer.post_attention_layernorm, nn.LayerNorm)
        self.assertIsInstance(layer_norm_model.norm, nn.LayerNorm)
        self.assertIsNone(cast(nn.LayerNorm, layer.self_attn.q_norm).bias)

        gelu_model = Olmo2ForCausalLM(config, variant="gelu")
        gelu_layer = cast(Olmo2DecoderLayer, gelu_model.layers[0])
        self.assertIsInstance(gelu_layer.mlp, GELUMLP)
        self.assertFalse(hasattr(gelu_layer.mlp, "gate_proj"))

        mha_model = Olmo2ForCausalLM(config, variant="mha")
        mha_layer = cast(Olmo2DecoderLayer, mha_model.layers[0])
        self.assertEqual(
            mha_layer.self_attn.k_proj.out_features,
            config.hidden_size,
        )
        self.assertEqual(mha_layer.self_attn.num_key_value_heads, 4)

        no_qk_model = Olmo2ForCausalLM(config, variant="without_qk_norm")
        no_qk_layer = cast(Olmo2DecoderLayer, no_qk_model.layers[0])
        no_qk_attention = no_qk_layer.self_attn
        self.assertIsInstance(no_qk_attention.q_norm, nn.Identity)
        self.assertIsInstance(no_qk_attention.k_norm, nn.Identity)

        tied_model = Olmo2ForCausalLM(config, variant="tied_embeddings")
        self.assertEqual(
            tied_model.embed_tokens.weight.data_ptr(),
            tied_model.lm_head.weight.data_ptr(),
        )
        self.assertLess(
            tied_model.trainable_parameter_count(),
            Olmo2ForCausalLM(config).trainable_parameter_count(),
        )

        absolute_model = Olmo2ForCausalLM(config, variant="learned_absolute_positions")
        self.assertIsNone(absolute_model.rotary_embedding)
        self.assertIsNotNone(absolute_model.position_embeddings)

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dense model variant"):
            Olmo2ForCausalLM(
                config=tiny_config(), variant=cast(DenseModelVariant, "unknown")
            )

    def test_baseline_norms_remain_rmsnorm(self) -> None:
        model = Olmo2ForCausalLM(tiny_config())
        layer = cast(Olmo2DecoderLayer, model.layers[0])
        attention = layer.self_attn
        self.assertIsInstance(attention.q_norm, RMSNorm)
        self.assertIsInstance(attention.k_norm, RMSNorm)


if __name__ == "__main__":
    unittest.main()
