from __future__ import annotations

import json
import unittest
from typing import cast

import torch
from pydantic import ValidationError
from torch import Tensor, nn
from torch.nn import functional as F

from lm_from_zero.models.config import Olmo2Config, fixed_special_tokens_hash
from lm_from_zero.models.interfaces import LayerKVCache
from lm_from_zero.models.olmo2 import (
    Olmo2Attention,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    RMSNorm,
    SwiGLU,
    cache_sequence_length,
)


def tiny_config(**updates: object) -> Olmo2Config:
    values: dict[str, object] = {
        "model_name": "test-olmo2",
        "tokenizer_hash": "0" * 64,
        "vocab_size": 272,
        "num_hidden_layers": 2,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 16,
    }
    values.update(updates)
    return Olmo2Config.model_validate(values)


class _ConstantAttention(nn.Module):
    def forward(
        self,
        hidden_states: Tensor,
        cosine: Tensor,
        sine: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerKVCache | None]:
        del cosine, sine, attention_mask, past, use_cache
        return torch.full_like(hidden_states, 2.0), None


class _ConstantMLP(nn.Module):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return torch.full_like(hidden_states, 4.0)


class Olmo2Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1337)

    def test_pinned_20m_parameter_count_is_exact(self) -> None:
        config = Olmo2Config(tokenizer_hash="a" * 64)
        breakdown = config.parameter_breakdown()
        model = Olmo2ForCausalLM(config)

        self.assertEqual(breakdown.total, 20_159_104)
        self.assertEqual(model.trainable_parameter_count(), breakdown.total)
        self.assertEqual(
            breakdown.model_dump(),
            {
                "token_embeddings": 6_144_000,
                "attention_projections": 1_966_080,
                "mlp_projections": 5_898_240,
                "normalization_scales": 6_784,
                "output_head": 6_144_000,
                "total": 20_159_104,
            },
        )

    def test_config_hash_and_flops_are_deterministic(self) -> None:
        config = tiny_config()
        restored = Olmo2Config.model_validate_json(config.canonical_json())
        estimate = config.forward_flops(8)

        self.assertEqual(config, restored)
        self.assertEqual(config.config_hash, restored.config_hash)
        self.assertEqual(json.loads(config.canonical_json())["architecture"], "olmo2")
        self.assertEqual(
            estimate.total_flops_per_token,
            estimate.projection_flops_per_token + estimate.attention_flops_per_token,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            config.forward_flops(17)

    def test_config_rejects_unknown_or_incompatible_values(self) -> None:
        with self.assertRaises(ValidationError):
            tiny_config(unknown_field=True)
        with self.assertRaisesRegex(ValidationError, "hidden_size"):
            tiny_config(hidden_size=30)
        with self.assertRaisesRegex(ValidationError, "key_value_heads"):
            tiny_config(num_key_value_heads=3)
        with self.assertRaisesRegex(ValidationError, "even"):
            tiny_config(hidden_size=20, num_attention_heads=4)
        with self.assertRaisesRegex(ValidationError, "special-token hash"):
            tiny_config(special_tokens_hash="f" * 64)
        self.assertEqual(fixed_special_tokens_hash(), tiny_config().special_tokens_hash)

    def test_model_is_bias_free_untied_and_zeros_padding_embedding(self) -> None:
        model = Olmo2ForCausalLM(tiny_config())
        first_layer = cast(Olmo2DecoderLayer, model.layers[0])
        linear_layers = [
            module for module in model.modules() if isinstance(module, nn.Linear)
        ]

        self.assertTrue(linear_layers)
        self.assertTrue(all(module.bias is None for module in linear_layers))
        self.assertNotEqual(
            model.embed_tokens.weight.data_ptr(), model.lm_head.weight.data_ptr()
        )
        torch.testing.assert_close(
            model.embed_tokens.weight[0],
            torch.zeros_like(model.embed_tokens.weight[0]),
        )
        self.assertEqual(first_layer.self_attn.q_norm.weight.shape, (32,))
        self.assertEqual(first_layer.self_attn.k_norm.weight.shape, (16,))

    def test_forward_loss_and_gradients(self) -> None:
        config = tiny_config()
        model = Olmo2ForCausalLM(config)
        first_layer = cast(Olmo2DecoderLayer, model.layers[0])
        input_ids = torch.randint(8, config.vocab_size, (2, 7))
        labels = input_ids.clone()
        labels[0, 3] = -100

        output = model(input_ids, labels=labels)
        self.assertEqual(output.logits.shape, (2, 7, config.vocab_size))
        self.assertIsNotNone(output.loss)
        loss = cast(Tensor, output.loss)
        expected = F.cross_entropy(
            output.logits[:, :-1].reshape(-1, config.vocab_size).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        torch.testing.assert_close(loss, expected)

        torch.autograd.backward(loss)
        gradients = (
            first_layer.self_attn.q_proj.weight.grad,
            first_layer.mlp.gate_proj.weight.grad,
            model.lm_head.weight.grad,
        )
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(
                bool(torch.isfinite(cast(Tensor, gradient)).all())
                for gradient in gradients
            )
        )

    def test_all_ignored_or_single_token_loss_is_differentiable_zero(self) -> None:
        model = Olmo2ForCausalLM(tiny_config())
        for input_ids, labels in (
            (torch.tensor([[8, 9]]), torch.full((1, 2), -100)),
            (torch.tensor([[8]]), torch.tensor([[8]])),
        ):
            model.zero_grad(set_to_none=True)
            output = model(input_ids, labels=labels)
            loss = cast(Tensor, output.loss)
            self.assertEqual(float(loss.detach()), 0.0)
            torch.autograd.backward(loss)
            self.assertIsNotNone(model.lm_head.weight.grad)

    def test_causal_logits_do_not_depend_on_future_tokens(self) -> None:
        model = Olmo2ForCausalLM(tiny_config()).eval()
        first = torch.tensor([[8, 9, 10, 11, 12, 13]])
        second = torch.tensor([[8, 9, 10, 101, 102, 103]])

        with torch.no_grad():
            first_logits = model(first).logits[:, :3]
            second_logits = model(second).logits[:, :3]

        torch.testing.assert_close(first_logits, second_logits)

    def test_gqa_qk_normalization_and_sdpa_match_manual_reference(self) -> None:
        config = tiny_config(num_hidden_layers=1)
        attention = Olmo2Attention(config).eval()
        hidden_states = torch.randn(2, 5, config.hidden_size)
        cosine = torch.ones(2, 5, config.head_dim)
        sine = torch.zeros_like(cosine)

        raw_query = attention.q_proj(hidden_states)
        raw_key = attention.k_proj(hidden_states)
        normalized_query = attention.q_norm(raw_query)
        normalized_key = attention.k_norm(raw_key)
        expected_query = raw_query.float() * torch.rsqrt(
            raw_query.float().square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
        )
        expected_key = raw_key.float() * torch.rsqrt(
            raw_key.float().square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
        )
        torch.testing.assert_close(normalized_query, expected_query)
        torch.testing.assert_close(normalized_key, expected_key)

        batch_size, sequence_length, _ = hidden_states.shape
        query = normalized_query.view(
            batch_size,
            sequence_length,
            config.num_attention_heads,
            config.head_dim,
        ).transpose(1, 2)
        key = normalized_key.view(
            batch_size,
            sequence_length,
            config.num_key_value_heads,
            config.head_dim,
        ).transpose(1, 2)
        value = (
            attention.v_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                config.num_key_value_heads,
                config.head_dim,
            )
            .transpose(1, 2)
        )
        groups = config.num_attention_heads // config.num_key_value_heads
        repeated_key = key.repeat_interleave(groups, dim=1)
        repeated_value = value.repeat_interleave(groups, dim=1)
        scores = torch.matmul(query, repeated_key.transpose(-1, -2))
        scores = scores / (config.head_dim**0.5)
        causal = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
        ).tril()
        probabilities = torch.softmax(
            scores.masked_fill(~causal, float("-inf")),
            dim=-1,
        )
        attended = torch.matmul(probabilities, repeated_value)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, config.hidden_size)
        )
        expected_output = attention.o_proj(attended)

        actual_output, cache = attention(
            hidden_states,
            cosine,
            sine,
            use_cache=True,
        )

        torch.testing.assert_close(
            actual_output,
            expected_output,
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertIsNotNone(cache)
        typed_cache = cast(LayerKVCache, cache)
        self.assertEqual(
            typed_cache.key.shape,
            (
                batch_size,
                config.num_key_value_heads,
                sequence_length,
                config.head_dim,
            ),
        )
        expanded_cache = typed_cache.key.repeat_interleave(groups, dim=1)
        for head in range(config.num_attention_heads):
            torch.testing.assert_close(
                expanded_cache[:, head],
                typed_cache.key[:, head // groups],
                rtol=0,
                atol=0,
            )

    def test_one_batch_overfits_to_approximately_zero_loss(self) -> None:
        config = tiny_config(
            vocab_size=32,
            num_hidden_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=32,
            max_position_embeddings=8,
        )
        model = Olmo2ForCausalLM(config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.05,
            weight_decay=0.0,
        )
        input_ids = torch.tensor([[8, 9, 10, 11, 12, 13]])
        final_loss = float("inf")

        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = cast(Tensor, model(input_ids, labels=input_ids).loss)
            torch.autograd.backward(loss)
            optimizer.step()
            final_loss = float(loss.detach())
            if final_loss < 0.01:
                break

        self.assertLess(final_loss, 0.01)

    def test_left_padding_mask_matches_unpadded_sequence(self) -> None:
        model = Olmo2ForCausalLM(tiny_config()).eval()
        plain = torch.tensor([[8, 9, 10]])
        padded = torch.tensor([[0, 0, 8, 9, 10]])
        attention_mask = torch.tensor([[False, False, True, True, True]])
        position_ids = torch.tensor([[0, 0, 0, 1, 2]])

        with torch.no_grad():
            plain_logits = model(plain).logits
            padded_logits = model(
                padded,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits[:, 2:]

        torch.testing.assert_close(plain_logits, padded_logits, atol=1e-6, rtol=1e-5)

    def test_token_by_token_cache_matches_full_forward(self) -> None:
        config = tiny_config()
        model = Olmo2ForCausalLM(config).eval()
        input_ids = torch.randint(8, config.vocab_size, (2, 6))

        with torch.no_grad():
            expected = model(input_ids).logits
            cache = None
            pieces: list[Tensor] = []
            for index in range(input_ids.shape[1]):
                output = model(
                    input_ids[:, index : index + 1],
                    cache=cache,
                    use_cache=True,
                )
                cache = output.cache
                pieces.append(output.logits)
            actual = torch.cat(pieces, dim=1)

        self.assertIsNotNone(cache)
        self.assertEqual(
            cache_sequence_length(cast(tuple[LayerKVCache, ...], cache)), 6
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_branch_outputs_are_normalized_before_residual_addition(self) -> None:
        config = tiny_config(num_hidden_layers=1)
        layer = Olmo2DecoderLayer(config)
        layer.self_attn = cast(Olmo2Attention, _ConstantAttention())
        layer.mlp = cast(SwiGLU, _ConstantMLP())
        hidden_states = torch.zeros(1, 2, config.hidden_size)
        positions = torch.zeros(1, 2, config.head_dim)

        output, _ = layer(hidden_states, positions, positions)

        expected = torch.full_like(output, 2.0)
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)

    def test_rms_norm_preserves_input_dtype(self) -> None:
        norm = RMSNorm(8, 1e-5)
        values = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        output = norm(values)

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_invalid_runtime_inputs_and_cache_are_rejected(self) -> None:
        config = tiny_config()
        model = Olmo2ForCausalLM(config)
        valid = torch.tensor([[8, 9]])
        invalid_cache = (LayerKVCache(torch.zeros(1), torch.zeros(1)),)

        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.tensor([8, 9]))
        with self.assertRaisesRegex(ValueError, "torch.long"):
            model(valid.float())
        with self.assertRaisesRegex(ValueError, "outside"):
            model(torch.tensor([[config.vocab_size]]))
        with self.assertRaisesRegex(ValueError, "position_ids"):
            model(valid, position_ids=torch.tensor([[0]]))
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            model(valid, attention_mask=torch.ones(1, 1))
        with self.assertRaisesRegex(ValueError, "layer count"):
            model(valid, cache=invalid_cache)

        too_long = torch.ones(1, config.max_position_embeddings + 1, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "context length"):
            model(too_long)

    def test_corrupt_layer_cache_is_rejected(self) -> None:
        model = Olmo2ForCausalLM(tiny_config(num_hidden_layers=1))
        wrong_rank = LayerKVCache(torch.zeros(1), torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "rank four"):
            model(torch.tensor([[8]]), cache=(wrong_rank,))

        different_lengths = (
            LayerKVCache(
                torch.zeros(1, 2, 1, 8),
                torch.zeros(1, 2, 1, 8),
            ),
            LayerKVCache(
                torch.zeros(1, 2, 2, 8),
                torch.zeros(1, 2, 2, 8),
            ),
        )
        with self.assertRaisesRegex(ValueError, "different sequence lengths"):
            cache_sequence_length(different_lengths)


if __name__ == "__main__":
    unittest.main()
