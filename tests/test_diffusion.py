from __future__ import annotations

import json
import unittest
from dataclasses import fields
from typing import cast

import torch
from pydantic import ValidationError
from torch import Tensor, nn
from torch.nn import functional as F

from lm_from_zero.models.config import (
    MaskedDiffusionConfig,
    fixed_special_tokens_hash,
)
from lm_from_zero.models.diffusion import (
    DiffusionCorruptionBatch,
    MaskedDiffusionForMaskedLM,
    base_pretraining_eligible_mask,
    corrupt_for_diffusion,
    masked_diffusion_loss,
)


def tiny_config(**updates: object) -> MaskedDiffusionConfig:
    values: dict[str, object] = {
        "model_name": "test-diffusion",
        "tokenizer_hash": "0" * 64,
        "vocab_size": 32,
        "num_hidden_layers": 2,
        "hidden_size": 24,
        "num_attention_heads": 4,
        "intermediate_size": 48,
        "max_position_embeddings": 16,
    }
    values.update(updates)
    return MaskedDiffusionConfig.model_validate(values)


class MaskedDiffusionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1337)

    def test_pinned_parameter_count_and_flops_are_exact(self) -> None:
        config = MaskedDiffusionConfig(tokenizer_hash="a" * 64)
        breakdown = config.parameter_breakdown()
        model = MaskedDiffusionForMaskedLM(config)

        self.assertEqual(breakdown.total, 19_959_168)
        self.assertEqual(model.trainable_parameter_count(), breakdown.total)
        self.assertEqual(
            breakdown.model_dump(),
            {
                "token_embeddings": 6_144_000,
                "attention_projections": 2_359_296,
                "mlp_projections": 5_308_416,
                "normalization_scales": 3_456,
                "output_head": 6_144_000,
                "total": 19_959_168,
            },
        )
        estimate = config.forward_flops(257)
        self.assertEqual(
            estimate.total_flops_per_token,
            estimate.projection_flops_per_token + estimate.attention_flops_per_token,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            config.forward_flops(1_025)

    def test_config_round_trip_and_invalid_values(self) -> None:
        config = tiny_config()
        restored = MaskedDiffusionConfig.model_validate_json(config.canonical_json())

        self.assertEqual(config, restored)
        self.assertEqual(config.config_hash, restored.config_hash)
        self.assertEqual(
            json.loads(config.canonical_json())["architecture"],
            "masked_diffusion",
        )
        self.assertFalse(config.time_conditioning)
        self.assertEqual(config.special_tokens_hash, fixed_special_tokens_hash())
        with self.assertRaises(ValidationError):
            tiny_config(unknown=True)
        with self.assertRaisesRegex(ValidationError, "divisible"):
            tiny_config(hidden_size=22)
        with self.assertRaisesRegex(ValidationError, "even"):
            tiny_config(hidden_size=20, num_attention_heads=4)
        with self.assertRaisesRegex(ValidationError, "special-token hash"):
            tiny_config(special_tokens_hash="f" * 64)

    def test_corruption_is_deterministic_and_protects_bos_padding(self) -> None:
        input_ids = torch.tensor(
            [
                [1, 8, 9, 2, 0],
                [1, 10, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ]
        )
        attention_mask = input_ids != 0
        eligible = base_pretraining_eligible_mask(input_ids, attention_mask)
        first = corrupt_for_diffusion(
            input_ids,
            eligible,
            mask_token_id=7,
            generator=torch.Generator().manual_seed(2027),
        )
        second = corrupt_for_diffusion(
            input_ids,
            eligible,
            mask_token_id=7,
            generator=torch.Generator().manual_seed(2027),
        )

        self.assertEqual(
            [field.name for field in fields(DiffusionCorruptionBatch)],
            ["input_ids", "labels", "eligible_mask", "corrupted_mask", "time"],
        )
        for field in fields(DiffusionCorruptionBatch):
            torch.testing.assert_close(
                cast(Tensor, getattr(first, field.name)),
                cast(Tensor, getattr(second, field.name)),
            )
        self.assertFalse(bool(first.corrupted_mask[:, 0].any()))
        self.assertFalse(bool(first.corrupted_mask[input_ids == 0].any()))
        self.assertTrue(bool(first.corrupted_mask[0].any()))
        self.assertTrue(bool(first.corrupted_mask[1].any()))
        self.assertFalse(bool(first.corrupted_mask[2].any()))
        self.assertTrue(bool(first.eligible_mask[0, 3]))
        self.assertEqual(
            torch.unique(first.input_ids[first.corrupted_mask]).tolist(), [7]
        )

    def test_corruption_matches_uniform_time_mask_rate(self) -> None:
        input_ids = torch.full((2_048, 128), 8, dtype=torch.long)
        eligible = torch.ones_like(input_ids, dtype=torch.bool)
        batch = corrupt_for_diffusion(
            input_ids,
            eligible,
            mask_token_id=7,
            epsilon=1e-3,
            generator=torch.Generator().manual_seed(3407),
        )

        expected_mean = (1.0 + 1e-3) / 2
        self.assertAlmostEqual(float(batch.time.mean()), expected_mean, delta=0.02)
        self.assertAlmostEqual(
            float(batch.corrupted_mask.float().mean()),
            expected_mean,
            delta=0.02,
        )
        self.assertTrue(bool(batch.corrupted_mask.any(dim=1).all()))

    def test_weighted_loss_matches_direct_equation_and_gradients(self) -> None:
        logits = torch.tensor(
            [
                [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [0.0, 2.0, 1.0]],
            ],
            requires_grad=True,
        )
        labels = torch.tensor([[0, -100, 2], [-100, 0, -100]])
        eligible = torch.tensor(
            [[True, True, True], [True, True, False]],
        )
        time = torch.tensor([0.5, 0.25])
        loss = masked_diffusion_loss(logits, labels, eligible, time)
        raw = F.cross_entropy(
            logits.reshape(-1, 3),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(2, 3)
        expected = torch.stack(
            (
                (raw[0, 0] + raw[0, 2]) / time[0] / 3,
                raw[1, 1] / time[1] / 2,
            )
        ).mean()

        torch.testing.assert_close(loss, expected)
        torch.autograd.backward(loss)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(cast(Tensor, logits.grad)).all()))

    def test_loss_rejects_contract_violations_and_handles_empty_batch(self) -> None:
        logits = torch.randn(1, 2, 8, requires_grad=True)
        empty = masked_diffusion_loss(
            logits,
            torch.full((1, 2), -100),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.tensor([0.5]),
        )
        self.assertEqual(float(empty.detach()), 0.0)
        torch.autograd.backward(empty)
        self.assertIsNotNone(logits.grad)

        with self.assertRaisesRegex(ValueError, "subset"):
            masked_diffusion_loss(
                logits,
                torch.tensor([[1, -100]]),
                torch.zeros(1, 2, dtype=torch.bool),
                torch.tensor([0.5]),
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            masked_diffusion_loss(
                logits,
                torch.full((1, 2), -100),
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([0.5]),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            masked_diffusion_loss(
                logits,
                torch.tensor([[1, -100]]),
                torch.tensor([[True, False]]),
                torch.tensor([float("nan")]),
            )

    def test_bidirectional_context_and_padding_mask(self) -> None:
        model = MaskedDiffusionForMaskedLM(tiny_config()).eval()
        first = torch.tensor([[8, 7, 9, 10]])
        second = torch.tensor([[8, 7, 9, 11]])
        padded = torch.tensor([[8, 7, 9, 10, 12]])
        padded_mask = torch.tensor([[True, True, True, True, False]])

        with torch.no_grad():
            first_logits = model(first).logits
            second_logits = model(second).logits
            padded_logits = model(padded, attention_mask=padded_mask).logits[:, :4]

        self.assertFalse(torch.allclose(first_logits[:, 0], second_logits[:, 0]))
        torch.testing.assert_close(first_logits, padded_logits, atol=1e-6, rtol=1e-5)

    def test_forward_objective_has_finite_gradients_and_no_bias_or_tying(self) -> None:
        config = tiny_config()
        model = MaskedDiffusionForMaskedLM(config)
        original = torch.tensor([[1, 8, 9, 2], [1, 10, 11, 2]])
        attention_mask = torch.ones_like(original, dtype=torch.bool)
        eligible = base_pretraining_eligible_mask(original, attention_mask)
        batch = corrupt_for_diffusion(
            original,
            eligible,
            mask_token_id=config.mask_token_id,
            epsilon=config.corruption_epsilon,
            generator=torch.Generator().manual_seed(1337),
        )
        output = model(
            batch.input_ids,
            attention_mask=attention_mask,
            labels=batch.labels,
            eligible_mask=batch.eligible_mask,
            time=batch.time,
        )

        self.assertEqual(output.logits.shape, (2, 4, config.vocab_size))
        loss = cast(Tensor, output.loss)
        self.assertTrue(bool(torch.isfinite(loss)))
        torch.autograd.backward(loss)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(
                bool(torch.isfinite(cast(Tensor, gradient)).all())
                for gradient in gradients
            )
        )
        linear_layers = [
            module for module in model.modules() if isinstance(module, nn.Linear)
        ]
        self.assertTrue(linear_layers)
        self.assertTrue(all(module.bias is None for module in linear_layers))
        self.assertNotEqual(
            model.embed_tokens.weight.data_ptr(),
            model.lm_head.weight.data_ptr(),
        )
        torch.testing.assert_close(
            model.embed_tokens.weight[config.pad_token_id],
            torch.zeros_like(model.embed_tokens.weight[config.pad_token_id]),
        )

    def test_linear_cross_entropy_objective_matches_full_logits(self) -> None:
        config = tiny_config(num_hidden_layers=1)
        model = MaskedDiffusionForMaskedLM(config)
        original = torch.tensor([[1, 8, 9, 2], [1, 10, 11, 2]])
        eligible = base_pretraining_eligible_mask(
            original, torch.ones_like(original, dtype=torch.bool)
        )
        batch = corrupt_for_diffusion(
            original,
            eligible,
            mask_token_id=config.mask_token_id,
            generator=torch.Generator().manual_seed(1337),
        )
        arguments = {
            "labels": batch.labels,
            "eligible_mask": batch.eligible_mask,
            "time": batch.time,
        }

        full = cast(Tensor, model(batch.input_ids, **arguments).loss)
        torch.autograd.backward(full)
        full_gradients = {
            name: cast(Tensor, parameter.grad).clone()
            for name, parameter in model.named_parameters()
        }
        model.zero_grad(set_to_none=True)
        output = model(
            batch.input_ids,
            **arguments,
            loss_backend="linear",
            loss_only=True,
        )
        linear = cast(Tensor, output.loss)
        torch.autograd.backward(linear)

        self.assertEqual(output.logits.numel(), 0)
        torch.testing.assert_close(linear, full, atol=2e-6, rtol=2e-6)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                cast(Tensor, parameter.grad),
                full_gradients[name],
                atol=2e-6,
                rtol=2e-5,
            )

    def test_invalid_inputs_and_partial_objective_are_rejected(self) -> None:
        model = MaskedDiffusionForMaskedLM(tiny_config())
        with self.assertRaisesRegex(ValueError, "torch.long"):
            model(torch.ones(1, 2))
        with self.assertRaisesRegex(ValueError, "visible token"):
            model(
                torch.zeros(1, 2, dtype=torch.long),
                attention_mask=torch.zeros(1, 2, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "all required"):
            model(
                torch.tensor([[8, 7]]),
                labels=torch.tensor([[8, -100]]),
            )
        with self.assertRaisesRegex(ValueError, "eligible_mask"):
            corrupt_for_diffusion(
                torch.tensor([[8, 9]]),
                torch.ones(1, 3, dtype=torch.bool),
                mask_token_id=7,
            )


if __name__ == "__main__":
    unittest.main()
