from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from pydantic import ValidationError

from lm_from_zero.generation.causal import (
    DEFAULT_SUPPRESSED_TOKEN_IDS,
    CausalGenerationConfig,
    GenerationError,
    _select_next_tokens,
    append_generation_record,
    create_generation_record,
    generate_causal,
)
from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM


def _model() -> Olmo2ForCausalLM:
    torch.manual_seed(73)
    return Olmo2ForCausalLM(
        Olmo2Config(
            model_name="generation-test",
            tokenizer_hash="a" * 64,
            vocab_size=272,
            num_hidden_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=16,
        )
    ).eval()


class NativeCausalGenerationTests(unittest.TestCase):
    def test_greedy_cache_matches_full_forward_reference(self) -> None:
        model = _model()
        prompt = [1, 8, 9]
        config = CausalGenerationConfig(max_new_tokens=4)
        result = generate_causal(model, [prompt], config)

        reference = list(prompt)
        expected: list[int] = []
        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                logits = model(torch.tensor([reference])).logits[:, -1, :].clone()
                logits[:, list(DEFAULT_SUPPRESSED_TOKEN_IDS)] = -torch.inf
                token_id = int(logits.argmax(dim=-1).item())
                expected.append(token_id)
                reference.append(token_id)
                if token_id == model.config.eos_token_id:
                    break
        self.assertEqual(result.generated_token_ids, (tuple(expected),))
        self.assertEqual(result.model_forwards, len(expected))
        self.assertEqual(result.generated_token_count, len(expected))
        self.assertGreater(result.tokens_per_second, 0)
        self.assertFalse(model.training)

    def test_seeded_sampling_is_reproducible_and_suppresses_control_tokens(
        self,
    ) -> None:
        model = _model()
        config = CausalGenerationConfig(
            strategy="sample",
            max_new_tokens=5,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            seed=2027,
        )
        first = generate_causal(model, [[1, 8]], config)
        second = generate_causal(model, [[1, 8]], config)
        self.assertEqual(first.generated_token_ids, second.generated_token_ids)
        self.assertTrue(
            set(first.generated_token_ids[0]).isdisjoint(DEFAULT_SUPPRESSED_TOKEN_IDS)
        )

        logits = torch.zeros((1, model.config.vocab_size))
        logits[0, 0] = 100
        logits[0, 8] = 10
        generator = torch.Generator().manual_seed(1)
        selected = _select_next_tokens(
            logits,
            CausalGenerationConfig(
                strategy="sample",
                max_new_tokens=1,
                top_k=1,
            ),
            generator,
        )
        self.assertEqual(selected.item(), 8)
        raw = _select_next_tokens(
            logits,
            CausalGenerationConfig(
                strategy="sample",
                max_new_tokens=1,
                top_k=1,
                allow_raw_special_tokens=True,
            ),
            generator,
        )
        self.assertEqual(raw.item(), 0)

    def test_batched_eos_events_and_left_padded_cache(self) -> None:
        model = _model()
        events: list[tuple[int | None, ...]] = []
        selected = (
            torch.tensor([2, 8], dtype=torch.long),
            torch.tensor([2, 2], dtype=torch.long),
        )
        with patch(
            "lm_from_zero.generation.causal._select_next_tokens",
            side_effect=selected,
        ):
            result = generate_causal(
                model,
                [[1, 8, 9], [1]],
                CausalGenerationConfig(max_new_tokens=4),
                on_token=lambda event: events.append(event.token_ids),
            )
        self.assertEqual(result.generated_token_ids, ((2,), (8, 2)))
        self.assertEqual(result.stop_reasons, ("eos", "eos"))
        self.assertEqual(result.model_forwards, 2)
        self.assertEqual(events, [(2, 8), (None, 2)])

    def test_rejects_invalid_options_prompts_and_context(self) -> None:
        model = _model()
        with self.assertRaisesRegex(ValidationError, "strategy='sample'"):
            CausalGenerationConfig(temperature=0.5)
        with self.assertRaisesRegex(GenerationError, "at least one prompt"):
            generate_causal(model, [], CausalGenerationConfig())
        with self.assertRaisesRegex(GenerationError, "cannot be empty"):
            generate_causal(model, [[]], CausalGenerationConfig())
        with self.assertRaisesRegex(GenerationError, "outside"):
            generate_causal(model, [[272]], CausalGenerationConfig())
        with self.assertRaisesRegex(GenerationError, "exceeds model context"):
            generate_causal(
                model,
                [[1] * 15],
                CausalGenerationConfig(max_new_tokens=2),
            )
        with self.assertRaisesRegex(GenerationError, "top_k"):
            generate_causal(
                model,
                [[1]],
                CausalGenerationConfig(
                    strategy="sample",
                    max_new_tokens=1,
                    top_k=273,
                ),
            )

        training_model = _model().train()
        generate_causal(
            training_model,
            [[1]],
            CausalGenerationConfig(max_new_tokens=1),
        )
        self.assertTrue(training_model.training)

    def test_generation_record_is_canonical_append_only_evidence(self) -> None:
        model = _model()
        prompts = [[1, 8]]
        result = generate_causal(
            model,
            prompts,
            CausalGenerationConfig(max_new_tokens=1),
        )
        record = create_generation_record(
            result,
            prompts,
            model_config_sha256=model.config.config_hash,
            tokenizer_sha256=model.config.tokenizer_hash,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.jsonl"
            append_generation_record(path, record)
            append_generation_record(path, record)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], record.canonical_json())
        self.assertEqual(
            json.loads(lines[0])["prompt_token_sha256"], record.prompt_token_sha256
        )


if __name__ == "__main__":
    unittest.main()
