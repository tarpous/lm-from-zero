from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as functional
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.post_training.chat import ChatMessage, Conversation
from lm_from_zero.post_training.dpo import PreferencePair
from lm_from_zero.post_training.dpo_train import (
    DPOTrainingError,
    _iter_preference_batches,
)
from lm_from_zero.post_training.preference_dataset import PreferenceRecord
from lm_from_zero.post_training.preference_evaluation import (
    PreferenceEvaluationConfig,
    _sequence_logprobs_and_policy_kl,
)
from lm_from_zero.tokenizer.bpe import ByteBPE


def _record(index: int) -> PreferenceRecord:
    return PreferenceRecord(
        source_index=index,
        prompt_id=f"prompt-{index}",
        pair=PreferencePair(
            prompt=Conversation(
                messages=(ChatMessage(role="user", content=f"question {index}"),)
            ),
            chosen=ChatMessage(role="assistant", content=f"good answer {index}"),
            rejected=ChatMessage(role="assistant", content=f"bad answer {index}"),
        ),
        score_chosen=8.0,
        score_rejected=1.0,
    )


class PreferenceEvaluationTests(unittest.TestCase):
    def test_sequence_scores_and_forward_kl_match_manual_calculation(self) -> None:
        policy_logits = torch.tensor(
            [[[0.5, -0.5, 0.0], [0.0, 0.3, -0.2], [0.2, -0.1, 0.4]]]
        )
        reference_logits = torch.tensor(
            [[[0.1, -0.2, 0.4], [0.4, -0.3, 0.1], [-0.2, 0.3, 0.0]]]
        )
        input_ids = torch.tensor([[0, 2, 1]], dtype=torch.long)
        response_mask = torch.tensor([[False, True, True]])

        (
            policy_sequence,
            reference_sequence,
            sequence_kl,
            response_token_counts,
        ) = _sequence_logprobs_and_policy_kl(
            policy_logits,
            reference_logits,
            input_ids,
            response_mask,
        )

        policy_log_distribution = functional.log_softmax(
            policy_logits[:, :-1].float(),
            dim=-1,
        )
        reference_log_distribution = functional.log_softmax(
            reference_logits[:, :-1].float(),
            dim=-1,
        )
        target_ids = input_ids[:, 1:]
        expected_policy = (
            policy_log_distribution.gather(
                -1,
                target_ids.unsqueeze(-1),
            )
            .squeeze(-1)
            .sum(dim=1)
        )
        expected_reference = (
            reference_log_distribution.gather(
                -1,
                target_ids.unsqueeze(-1),
            )
            .squeeze(-1)
            .sum(dim=1)
        )
        expected_kl = (
            (
                policy_log_distribution.exp()
                * (policy_log_distribution - reference_log_distribution)
            )
            .sum(dim=-1)
            .sum(dim=1)
        )

        torch.testing.assert_close(policy_sequence, expected_policy)
        torch.testing.assert_close(reference_sequence, expected_reference)
        torch.testing.assert_close(sequence_kl, expected_kl)
        torch.testing.assert_close(response_token_counts, torch.tensor([2]))

    def test_configuration_hash_is_stable_and_validated(self) -> None:
        first = PreferenceEvaluationConfig()
        second = PreferenceEvaluationConfig()
        self.assertEqual(first.config_sha256, second.config_sha256)
        with self.assertRaises(ValueError):
            PreferenceEvaluationConfig(batch_size=0)

    def test_bounded_batch_iterator_stops_at_the_requested_pair_count(self) -> None:
        with TemporaryDirectory() as directory:
            records_path = Path(directory) / "records.jsonl"
            records_path.write_text(
                "".join(
                    record.model_dump_json() + "\n"
                    for record in (_record(index) for index in range(4))
                ),
                encoding="utf-8",
            )
            batches = list(
                _iter_preference_batches(
                    records_path,
                    ByteBPE(),
                    expected_records=2,
                    batch_size=2,
                    bucket_size=4,
                    max_length=1_024,
                    max_records=2,
                )
            )
            self.assertEqual(sum(len(batch.indices) for batch in batches), 2)
            with self.assertRaisesRegex(DPOTrainingError, "maximum"):
                list(
                    _iter_preference_batches(
                        records_path,
                        ByteBPE(),
                        expected_records=1,
                        batch_size=1,
                        bucket_size=1,
                        max_length=1_024,
                        max_records=0,
                    )
                )

    def test_evaluation_cli_forwards_a_bounded_smoke_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            holdout_manifest = Path(directory) / "manifest.json"
            holdout_manifest.write_text("{}", encoding="utf-8")
            reported = Mock()
            reported.model_dump_json.return_value = "{}"
            with patch(
                "lm_from_zero.post_training.preference_evaluation."
                "run_dpo_heldout_evaluation",
                return_value=reported,
            ) as evaluate:
                result = CliRunner().invoke(
                    app,
                    [
                        "evaluate-dpo-holdout",
                        str(holdout_manifest),
                        "--batch-size",
                        "1",
                        "--bucket-size",
                        "8",
                        "--max-pairs",
                        "8",
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(result.stdout.strip(), "{}")
            self.assertEqual(
                evaluate.call_args.kwargs["config"],
                PreferenceEvaluationConfig(batch_size=1, bucket_size=8, max_pairs=8),
            )


if __name__ == "__main__":
    unittest.main()
