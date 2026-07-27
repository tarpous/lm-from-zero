from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import torch
from pydantic import ValidationError
from torch import Tensor, nn
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.generation.diffusion import (
    DiffusionGenerationConfig,
    DiffusionGenerationError,
    DiffusionGenerationEvent,
    DiffusionGenerationResult,
    append_diffusion_generation_record,
    create_diffusion_generation_record,
    generate_diffusion,
)
from lm_from_zero.models import (
    MaskedDiffusionConfig,
    MaskedDiffusionForMaskedLM,
    MaskedDiffusionOutput,
)


def tiny_config(**updates: object) -> MaskedDiffusionConfig:
    values: dict[str, object] = {
        "model_name": "generation-diffusion-test",
        "tokenizer_hash": "0" * 64,
        "vocab_size": 32,
        "num_hidden_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 16,
    }
    values.update(updates)
    return MaskedDiffusionConfig.model_validate(values)


class _ScriptedDenoiser(nn.Module):
    def __init__(
        self,
        config: MaskedDiffusionConfig,
        *,
        uniform_logits: bool = False,
        eos_positions: set[int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.anchor = nn.Parameter(torch.zeros(()))
        self.uniform_logits = uniform_logits
        self.eos_positions = set() if eos_positions is None else eos_positions
        self.inputs: list[Tensor] = []

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        **kwargs: object,
    ) -> MaskedDiffusionOutput:
        del attention_mask, kwargs
        self.inputs.append(input_ids.detach().clone())
        logits = torch.zeros(
            *input_ids.shape,
            self.config.vocab_size,
            device=input_ids.device,
        )
        if not self.uniform_logits:
            logits.fill_(-8)
            for position in range(input_ids.shape[1]):
                token_id = (
                    self.config.eos_token_id
                    if position in self.eos_positions
                    else 8 + position % (self.config.vocab_size - 8)
                )
                logits[:, position, token_id] = 2.0 + position
        return MaskedDiffusionOutput(logits=logits)


class DiffusionGenerationTests(unittest.TestCase):
    def test_greedy_linear_reveals_are_deterministic_and_prompt_is_immutable(
        self,
    ) -> None:
        model = _ScriptedDenoiser(tiny_config())
        prompts = [[1, 8], [1, 9, 10]]
        config = DiffusionGenerationConfig(
            response_length=4,
            diffusion_steps=4,
        )
        events: list[DiffusionGenerationEvent] = []
        result = generate_diffusion(
            cast(MaskedDiffusionForMaskedLM, model),
            prompts,
            config,
            on_step=events.append,
        )

        self.assertEqual(result.model_forwards, 4)
        self.assertEqual(result.diffusion_steps, 4)
        self.assertEqual(result.stop_reasons, ("canvas_complete", "canvas_complete"))
        self.assertEqual(
            [event.remaining_masks for event in events],
            [(3, 3), (2, 2), (1, 1), (0, 0)],
        )
        for recorded in model.inputs:
            self.assertEqual(recorded[0, :2].tolist(), prompts[0])
            self.assertEqual(recorded[1, :3].tolist(), prompts[1])
        self.assertNotIn(
            model.config.mask_token_id,
            result.generated_token_ids[0],
        )

    def test_cosine_reduced_steps_fill_canvas_and_eos_truncates(self) -> None:
        prompt = [1, 8]
        model = _ScriptedDenoiser(
            tiny_config(),
            eos_positions={len(prompt)},
        )
        result = generate_diffusion(
            cast(MaskedDiffusionForMaskedLM, model),
            [prompt],
            DiffusionGenerationConfig(
                response_length=5,
                diffusion_steps=2,
                reveal_schedule="cosine",
            ),
        )

        self.assertEqual(result.model_forwards, 2)
        self.assertEqual(result.stop_reasons, ("eos",))
        self.assertEqual(result.generated_token_ids, ((model.config.eos_token_id,),))

    def test_temperature_sampling_is_seeded(self) -> None:
        prompts = [[1, 8]]

        def run(seed: int) -> tuple[int, ...]:
            model = _ScriptedDenoiser(tiny_config(), uniform_logits=True)
            result = generate_diffusion(
                cast(MaskedDiffusionForMaskedLM, model),
                prompts,
                DiffusionGenerationConfig(
                    strategy="sample",
                    response_length=8,
                    diffusion_steps=3,
                    temperature=0.8,
                    seed=seed,
                ),
            )
            return result.generated_token_ids[0]

        self.assertEqual(run(42), run(42))
        self.assertNotEqual(run(42), run(43))

    def test_low_confidence_remasking_reopens_revealed_positions(self) -> None:
        prompt = [1, 8]
        model = _ScriptedDenoiser(tiny_config())
        events: list[DiffusionGenerationEvent] = []
        result = generate_diffusion(
            cast(MaskedDiffusionForMaskedLM, model),
            [prompt],
            DiffusionGenerationConfig(
                response_length=4,
                diffusion_steps=4,
                remask_strategy="low_confidence",
                remask_fraction=0.5,
            ),
            on_step=events.append,
        )

        response_slice = slice(len(prompt), len(prompt) + 4)
        step_three_input = model.inputs[2][0, response_slice]
        self.assertEqual(
            int((step_three_input == model.config.mask_token_id).sum()),
            3,
        )
        self.assertTrue(events[2].remasked_positions[0])
        self.assertEqual(events[-1].remaining_masks, (0,))
        self.assertNotIn(model.config.mask_token_id, result.generated_token_ids[0])

    def test_real_model_terminates_and_restores_training_mode(self) -> None:
        model = MaskedDiffusionForMaskedLM(tiny_config())
        model.train()
        first = generate_diffusion(
            model,
            [[1, 8, 9]],
            DiffusionGenerationConfig(response_length=3, seed=7),
        )
        second = generate_diffusion(
            model,
            [[1, 8, 9]],
            DiffusionGenerationConfig(response_length=3, seed=7),
        )

        self.assertTrue(model.training)
        self.assertEqual(first.generated_token_ids, second.generated_token_ids)
        self.assertEqual(first.model_forwards, 3)

    def test_record_append_and_invalid_contracts(self) -> None:
        model = _ScriptedDenoiser(tiny_config())
        prompts = [[1, 8]]
        result = generate_diffusion(
            cast(MaskedDiffusionForMaskedLM, model),
            prompts,
            DiffusionGenerationConfig(response_length=2),
        )
        record = create_diffusion_generation_record(
            result,
            prompts,
            model_config_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generation.jsonl"
            append_diffusion_generation_record(output, record)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                record.canonical_json() + "\n",
            )

        with self.assertRaises(ValidationError):
            DiffusionGenerationConfig(
                strategy="greedy",
                temperature=0.5,
            )
        with self.assertRaises(ValidationError):
            DiffusionGenerationConfig(
                remask_strategy="low_confidence",
                remask_fraction=0,
            )
        with self.assertRaisesRegex(DiffusionGenerationError, "at least one"):
            generate_diffusion(
                cast(MaskedDiffusionForMaskedLM, model),
                [],
                DiffusionGenerationConfig(response_length=2),
            )
        with self.assertRaisesRegex(DiffusionGenerationError, "context"):
            generate_diffusion(
                cast(MaskedDiffusionForMaskedLM, model),
                [[1] * 15],
                DiffusionGenerationConfig(response_length=2),
            )

    def test_cli_loads_bound_checkpoint_and_dispatches_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            tokenizer_manifest = root / "training.json"
            tokenizer_manifest.write_text("{}")
            model_config = tiny_config(tokenizer_hash="a" * 64)
            binding = SimpleNamespace(
                architecture="masked_diffusion",
                resolved_model_config=model_config.model_dump(mode="json"),
                tokenizer_sha256="a" * 64,
            )
            manifest = SimpleNamespace(binding=binding)
            training = SimpleNamespace(
                status="complete",
                tokenizer_hash="a" * 64,
                tokenizer_file="tokenizer.json",
            )
            tokenizer = SimpleNamespace(
                model_hash="a" * 64,
                encode=Mock(return_value=[1, 8]),
                decode=Mock(return_value="finished"),
            )
            model = Mock()
            result = DiffusionGenerationResult(
                prompt_token_counts=(2,),
                response_canvas_length=2,
                generated_token_ids=((9, 2),),
                stop_reasons=("eos",),
                diffusion_steps=1,
                model_forwards=1,
                generated_token_count=2,
                elapsed_seconds=0.1,
                tokens_per_second=20,
            )
            sampler = Mock(return_value=result)
            append = Mock()
            output = root / "generation.jsonl"
            with (
                patch(
                    "lm_from_zero.training.validate_checkpoint",
                    return_value=manifest,
                ),
                patch("lm_from_zero.cli.load_training_manifest", return_value=training),
                patch(
                    "lm_from_zero.tokenizer.bpe.ByteBPE.load",
                    return_value=tokenizer,
                ),
                patch(
                    "lm_from_zero.models.MaskedDiffusionForMaskedLM",
                    return_value=model,
                ),
                patch("lm_from_zero.training.load_checkpoint_model"),
                patch(
                    "lm_from_zero.generation.generate_diffusion",
                    sampler,
                ),
                patch(
                    "lm_from_zero.generation.append_diffusion_generation_record",
                    append,
                ),
            ):
                invocation = CliRunner().invoke(
                    app,
                    [
                        "generate-diffusion",
                        str(checkpoint),
                        str(tokenizer_manifest),
                        "hello",
                        "--response-length",
                        "2",
                        "--diffusion-steps",
                        "1",
                        "--jsonl-output",
                        str(output),
                    ],
                )

            self.assertEqual(invocation.exit_code, 0, invocation.output)
            payload = json.loads(invocation.stdout)
            self.assertEqual(payload["generated_text"], "finished")
            self.assertEqual(payload["model_forwards"], 1)
            passed_config = sampler.call_args.args[2]
            self.assertEqual(passed_config.response_length, 2)
            self.assertEqual(passed_config.diffusion_steps, 1)
            append.assert_called_once()


if __name__ == "__main__":
    unittest.main()
