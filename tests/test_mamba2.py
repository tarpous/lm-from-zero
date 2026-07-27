from __future__ import annotations

import json
import unittest
from typing import cast

import torch
from pydantic import ValidationError
from torch import Tensor, nn
from torch.nn import functional as F

from lm_from_zero.models import (
    Mamba2Cache,
    Mamba2Config,
    Mamba2ForCausalLM,
    Mamba2LayerState,
)
from lm_from_zero.models.mamba2 import (
    GroupedGatedRMSNorm,
    Mamba2Block,
    Mamba2Mixer,
    ssd_chunked,
    ssd_quadratic_reference,
    ssd_sequential_reference,
)


def tiny_config(**updates: object) -> Mamba2Config:
    values: dict[str, object] = {
        "model_name": "test-mamba2",
        "tokenizer_hash": "0" * 64,
        "vocab_size": 64,
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "state_size": 4,
        "expand": 2,
        "head_dim": 8,
        "num_heads": 4,
        "num_groups": 2,
        "conv_kernel": 3,
        "chunk_size": 4,
        "max_position_embeddings": 32,
    }
    values.update(updates)
    return Mamba2Config.model_validate(values)


class Mamba2Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1337)

    def test_pinned_20m_parameter_count_is_exact(self) -> None:
        config = Mamba2Config(tokenizer_hash="a" * 64)
        breakdown = config.parameter_breakdown()
        model = Mamba2ForCausalLM(config)

        self.assertEqual(breakdown.total, 19_943_164)
        self.assertEqual(model.trainable_parameter_count(), breakdown.total)
        self.assertEqual(
            breakdown.model_dump(),
            {
                "token_embeddings": 6_144_000,
                "input_projections": 5_537_280,
                "causal_convolution": 44_800,
                "ssm_parameters": 252,
                "normalization_scales": 8_448,
                "output_projections": 2_064_384,
                "output_head": 6_144_000,
                "total": 19_943_164,
            },
        )

    def test_config_hash_flops_and_validation(self) -> None:
        config = tiny_config()
        restored = Mamba2Config.model_validate_json(config.canonical_json())
        estimate = config.forward_flops(17)

        self.assertEqual(config, restored)
        self.assertEqual(config.config_hash, restored.config_hash)
        self.assertEqual(json.loads(config.canonical_json())["architecture"], "mamba2")
        self.assertEqual(
            estimate.total_flops_per_token,
            estimate.projection_flops_per_token
            + estimate.convolution_flops_per_token
            + estimate.ssm_flops_per_token,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            config.forward_flops(33)
        with self.assertRaises(ValidationError):
            tiny_config(unknown=True)
        with self.assertRaisesRegex(ValidationError, "expanded hidden"):
            tiny_config(num_heads=3)
        with self.assertRaisesRegex(ValidationError, "num_heads"):
            tiny_config(num_heads=8, head_dim=4, num_groups=3)
        with self.assertRaisesRegex(ValidationError, "time_step"):
            tiny_config(time_step_min=0.2, time_step_max=0.1)
        with self.assertRaisesRegex(ValidationError, "a_init"):
            tiny_config(a_init_min=2.0, a_init_max=1.0)

    def test_sequential_quadratic_and_chunked_ssd_are_equivalent(self) -> None:
        batch, sequence, heads, head_dim = 2, 7, 4, 3
        groups, state_size = 2, 5
        x = torch.randn(batch, sequence, heads, head_dim, requires_grad=True)
        log_decay = -torch.rand(batch, sequence, heads)
        b = torch.randn(batch, sequence, groups, state_size)
        c = torch.randn(batch, sequence, groups, state_size)

        sequential, sequential_state = ssd_sequential_reference(
            x,
            log_decay,
            b,
            c,
        )
        quadratic = ssd_quadratic_reference(x, log_decay, b, c)
        chunked, chunked_state = ssd_chunked(
            x,
            log_decay,
            b,
            c,
            chunk_size=4,
        )

        torch.testing.assert_close(quadratic, sequential, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(chunked, sequential, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            chunked_state,
            sequential_state,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.autograd.backward(chunked.square().mean())
        self.assertIsNotNone(x.grad)
        self.assertTrue(bool(torch.isfinite(cast(Tensor, x.grad)).all()))

        initial = torch.randn(batch, heads, head_dim, state_size)
        expected, expected_state = ssd_sequential_reference(
            x.detach(),
            log_decay,
            b,
            c,
            initial_state=initial,
        )
        actual, actual_state = ssd_chunked(
            x.detach(),
            log_decay,
            b,
            c,
            chunk_size=3,
            initial_state=initial,
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)

    def test_gated_group_norm_matches_direct_equation(self) -> None:
        norm = GroupedGatedRMSNorm(12, groups=3, eps=1e-5)
        values = torch.randn(2, 4, 12)
        gate = torch.randn_like(values)
        actual = norm(values, gate)

        gated = values * F.silu(gate)
        grouped = gated.view(2, 4, 3, 4)
        expected = grouped * torch.rsqrt(
            grouped.square().mean(dim=-1, keepdim=True) + norm.eps
        )
        expected = expected.flatten(start_dim=-2) * norm.weight
        torch.testing.assert_close(actual, expected)

    def test_initialization_forward_loss_and_finite_long_state(self) -> None:
        config = tiny_config(num_hidden_layers=1)
        model = Mamba2ForCausalLM(config)
        block = cast(Mamba2Block, model.layers[0])
        mixer = block.mixer
        initialized_dt = F.softplus(mixer.dt_bias.detach())
        initialized_a = torch.exp(mixer.A_log.detach())

        self.assertEqual(mixer.dt_bias.dtype, torch.float32)
        self.assertEqual(mixer.A_log.dtype, torch.float32)
        self.assertTrue(bool(torch.all(initialized_dt >= config.time_step_min)))
        self.assertTrue(bool(torch.all(initialized_dt <= config.time_step_max)))
        self.assertTrue(bool(torch.all(initialized_a >= config.a_init_min)))
        self.assertTrue(bool(torch.all(initialized_a <= config.a_init_max)))
        self.assertNotEqual(
            model.embed_tokens.weight.data_ptr(),
            model.lm_head.weight.data_ptr(),
        )
        torch.testing.assert_close(
            model.embed_tokens.weight[config.pad_token_id],
            torch.zeros_like(model.embed_tokens.weight[config.pad_token_id]),
        )

        input_ids = torch.randint(8, config.vocab_size, (2, 31))
        output = model(input_ids, labels=input_ids, use_cache=True)
        self.assertEqual(output.logits.shape, (2, 31, config.vocab_size))
        self.assertIsNotNone(output.loss)
        self.assertIsInstance(output.cache, Mamba2Cache)
        cache = cast(Mamba2Cache, output.cache)
        self.assertTrue(
            all(bool(torch.isfinite(state.ssm).all()) for state in cache.layers)
        )
        torch.autograd.backward(cast(Tensor, output.loss))
        gradients = (
            mixer.in_proj.weight.grad,
            mixer.A_log.grad,
            mixer.dt_bias.grad,
            model.lm_head.weight.grad,
        )
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(
                bool(torch.isfinite(cast(Tensor, gradient)).all())
                for gradient in gradients
            )
        )

    def test_causal_logits_cache_and_state_shapes(self) -> None:
        config = tiny_config()
        model = Mamba2ForCausalLM(config).eval()
        first = torch.tensor([[8, 9, 10, 11, 12, 13]])
        second = torch.tensor([[8, 9, 10, 41, 42, 43]])

        with torch.no_grad():
            first_logits = model(first).logits
            second_logits = model(second).logits
            cache = None
            pieces: list[Tensor] = []
            for index in range(first.shape[1]):
                output = model(
                    first[:, index : index + 1],
                    cache=cache,
                    use_cache=True,
                )
                cache = cast(Mamba2Cache, output.cache)
                pieces.append(output.logits)
            cached_logits = torch.cat(pieces, dim=1)

        torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])
        torch.testing.assert_close(cached_logits, first_logits, atol=2e-5, rtol=2e-5)
        self.assertIsNotNone(cache)
        typed_cache = cast(Mamba2Cache, cache)
        self.assertEqual(typed_cache.sequence_length, first.shape[1])
        self.assertEqual(len(typed_cache.layers), config.num_hidden_layers)
        for state in typed_cache.layers:
            self.assertEqual(
                state.convolution.shape,
                (1, config.convolution_size, config.conv_kernel),
            )
            self.assertEqual(
                state.ssm.shape,
                (
                    1,
                    config.num_heads,
                    config.head_dim,
                    config.state_size,
                ),
            )

    def test_left_padding_matches_unpadded_and_right_padding_is_rejected(self) -> None:
        model = Mamba2ForCausalLM(tiny_config()).eval()
        plain = torch.tensor([[8, 9, 10]])
        padded = torch.tensor([[0, 0, 8, 9, 10]])
        left_mask = torch.tensor([[0, 0, 1, 1, 1]])

        with torch.no_grad():
            expected = model(plain).logits
            actual = model(padded, attention_mask=left_mask).logits[:, 2:]
            repeated = model(plain).logits

        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(repeated, expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "right padding"):
            model(
                torch.tensor([[8, 9, 0]]),
                attention_mask=torch.tensor([[1, 1, 0]]),
            )

    def test_invalid_runtime_inputs_and_cache_are_rejected(self) -> None:
        config = tiny_config(num_hidden_layers=1)
        model = Mamba2ForCausalLM(config)
        valid = torch.tensor([[8, 9]])
        bad_state = Mamba2LayerState(
            convolution=torch.zeros(1),
            ssm=torch.zeros(1),
        )

        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.tensor([8, 9]))
        with self.assertRaisesRegex(ValueError, "torch.long"):
            model(valid.float())
        with self.assertRaisesRegex(ValueError, "outside"):
            model(torch.tensor([[config.vocab_size]]))
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            model(valid, attention_mask=torch.ones(1, 1))
        with self.assertRaisesRegex(ValueError, "layer count"):
            model(
                valid,
                cache=Mamba2Cache(layers=(), sequence_length=1),
            )
        with self.assertRaisesRegex(ValueError, "convolution state"):
            model(
                torch.tensor([[8]]),
                cache=Mamba2Cache(layers=(bad_state,), sequence_length=1),
            )
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            values = torch.ones(1, 2, 1, 1)
            ssd_chunked(
                values,
                torch.zeros(1, 2, 1),
                values,
                values,
                chunk_size=0,
            )

    def test_model_is_bias_free_except_for_depthwise_convolution(self) -> None:
        model = Mamba2ForCausalLM(tiny_config())
        linear_layers = [
            module for module in model.modules() if isinstance(module, nn.Linear)
        ]
        mixers = [
            module for module in model.modules() if isinstance(module, Mamba2Mixer)
        ]

        self.assertTrue(linear_layers)
        self.assertTrue(all(module.bias is None for module in linear_layers))
        self.assertTrue(mixers)
        self.assertTrue(all(mixer.conv1d.bias is not None for mixer in mixers))


if __name__ == "__main__":
    unittest.main()
