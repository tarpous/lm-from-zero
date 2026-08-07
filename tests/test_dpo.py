from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from pydantic import ValidationError

from lm_from_zero.post_training import (
    ChatMessage,
    ChatRole,
    Conversation,
    DPOConfig,
    DPOFormatError,
    PreferencePair,
    ReferenceLogProbCacheIdentity,
    dpo_objective,
    masked_sequence_logprob,
    render_preference_pair,
)
from lm_from_zero.tokenizer.bpe import ByteBPE


def _pair(
    *,
    prompt: tuple[tuple[ChatRole, str], ...] = (("user", "question"),),
    chosen: str = "good answer",
    rejected: str = "bad answer",
) -> PreferencePair:
    return PreferencePair(
        prompt=Conversation(
            messages=tuple(
                ChatMessage(role=role, content=content) for role, content in prompt
            )
        ),
        chosen=ChatMessage(role="assistant", content=chosen),
        rejected=ChatMessage(role="assistant", content=rejected),
    )


class DPOTests(unittest.TestCase):
    def test_preference_pair_schema_and_defaults(self) -> None:
        pair = _pair()
        self.assertEqual(pair.format_version, 1)
        self.assertEqual(DPOConfig().beta, 0.1)
        self.assertEqual(DPOConfig().learning_rate, 5e-7)
        with self.assertRaisesRegex(ValueError, "must end with a user"):
            _pair(prompt=(("user", "question"), ("assistant", "prior")))
        with self.assertRaisesRegex(ValidationError, "assistant"):
            PreferencePair(
                prompt=Conversation(
                    messages=(ChatMessage(role="user", content="question"),)
                ),
                chosen=ChatMessage(role="user", content="wrong"),
                rejected=ChatMessage(role="assistant", content="answer"),
            )
        with self.assertRaisesRegex(ValueError, "must differ"):
            _pair(chosen="same", rejected="same")

    def test_rendering_masks_shared_prompt_and_truncates_deterministically(
        self,
    ) -> None:
        tokenizer = ByteBPE()
        rendered = render_preference_pair(_pair(), tokenizer)
        self.assertEqual(
            rendered.chosen.input_ids[: rendered.chosen.prompt_prefix_length],
            rendered.rejected.input_ids[: rendered.rejected.prompt_prefix_length],
        )
        self.assertEqual(
            rendered.chosen.prompt_prefix_length,
            rendered.rejected.prompt_prefix_length,
        )
        self.assertFalse(
            any(rendered.chosen.response_mask[: rendered.chosen.prompt_prefix_length])
        )
        self.assertTrue(
            all(rendered.chosen.response_mask[rendered.chosen.prompt_prefix_length :])
        )
        self.assertLessEqual(len(rendered.chosen.input_ids), 1_024)

        long_pair = _pair(
            prompt=(("user", "prompt " * 20),),
            chosen="chosen " * 20,
            rejected="rejected " * 20,
        )
        with self.assertRaisesRegex(DPOFormatError, "exceeds"):
            render_preference_pair(long_pair, tokenizer, max_length=32)
        truncated = render_preference_pair(
            long_pair, tokenizer, max_length=32, truncation="left"
        )
        self.assertTrue(truncated.chosen.truncated)
        self.assertTrue(truncated.rejected.truncated)
        self.assertLessEqual(len(truncated.chosen.input_ids), 32)
        self.assertLessEqual(len(truncated.rejected.input_ids), 32)
        self.assertEqual(
            truncated.chosen.input_ids[: truncated.chosen.prompt_prefix_length],
            truncated.rejected.input_ids[: truncated.rejected.prompt_prefix_length],
        )
        with self.assertRaisesRegex(DPOFormatError, "unsupported"):
            render_preference_pair(_pair(), tokenizer, truncation="middle")  # type: ignore[arg-type]

    def test_masked_sequence_logprob_matches_manual_response_sum(self) -> None:
        logits = torch.tensor(
            [[[0.3, 0.0, -0.3], [0.1, 0.2, 0.3], [0.6, 0.0, -0.2], [0.0, 0.5, 0.1]]],
            requires_grad=True,
        )
        input_ids = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
        response_mask = torch.tensor([[False, False, True, True]])
        actual = masked_sequence_logprob(logits, input_ids, response_mask)
        expected = (
            torch.stack(
                (
                    F.log_softmax(logits[0, 1].float(), dim=-1)[input_ids[0, 2]],
                    F.log_softmax(logits[0, 2].float(), dim=-1)[input_ids[0, 3]],
                )
            )
            .sum()
            .reshape(1)
        )
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()  # type: ignore[no-untyped-call]
        self.assertIsNotNone(logits.grad)
        with self.assertRaisesRegex(DPOFormatError, "shape"):
            masked_sequence_logprob(torch.randn(3, 4), input_ids, response_mask)
        with self.assertRaisesRegex(DPOFormatError, "torch.long"):
            masked_sequence_logprob(logits.detach(), input_ids.float(), response_mask)
        with self.assertRaisesRegex(DPOFormatError, "torch.bool"):
            masked_sequence_logprob(logits.detach(), input_ids, response_mask.long())
        with self.assertRaisesRegex(DPOFormatError, "every sequence"):
            masked_sequence_logprob(
                logits.detach(), input_ids, torch.zeros_like(response_mask)
            )

    def test_dpo_objective_matches_hand_calculation_and_validates_batches(self) -> None:
        policy_chosen = torch.tensor([-2.0], requires_grad=True)
        policy_rejected = torch.tensor([-3.0], requires_grad=True)
        reference_chosen = torch.tensor([-1.0])
        reference_rejected = torch.tensor([-1.5])
        output = dpo_objective(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=0.1,
        )
        expected_logits = torch.tensor([0.05])
        torch.testing.assert_close(output.logits, expected_logits)
        torch.testing.assert_close(output.loss, -F.logsigmoid(expected_logits).mean())
        torch.testing.assert_close(output.chosen_rewards, torch.tensor([-0.1]))
        torch.testing.assert_close(output.rejected_rewards, torch.tensor([-0.15]))
        output.loss.backward()  # type: ignore[no-untyped-call]
        self.assertIsNotNone(policy_chosen.grad)
        with self.assertRaisesRegex(DPOFormatError, "positive"):
            dpo_objective(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta=0,
            )
        with self.assertRaisesRegex(DPOFormatError, "one-dimensional"):
            dpo_objective(
                policy_chosen.unsqueeze(0),
                policy_rejected,
                reference_chosen,
                reference_rejected,
            )
        with self.assertRaisesRegex(DPOFormatError, "equal shapes"):
            dpo_objective(
                policy_chosen,
                torch.tensor([-3.0, -2.0]),
                reference_chosen,
                reference_rejected,
            )

    def test_reference_cache_identity_is_objective_independent_and_hash_bound(
        self,
    ) -> None:
        identity = ReferenceLogProbCacheIdentity(
            model_hash="a" * 64,
            tokenizer_hash="b" * 64,
            checkpoint_hash="c" * 64,
            template_hash="d" * 64,
            max_length=1_024,
        )
        changed = ReferenceLogProbCacheIdentity(
            model_hash="a" * 64,
            tokenizer_hash="b" * 64,
            checkpoint_hash="e" * 64,
            template_hash="d" * 64,
            max_length=1_024,
        )
        self.assertEqual(identity.format_version, 1)
        self.assertEqual(len(identity.cache_key), 64)
        self.assertNotEqual(identity.cache_key, changed.cache_key)
        self.assertNotIn("dpo", identity.canonical_json().lower())
        self.assertNotIn("apo", identity.canonical_json().lower())
        with self.assertRaisesRegex(ValidationError, "Extra inputs are not permitted"):
            ReferenceLogProbCacheIdentity(  # type: ignore[call-arg]
                model_hash="a" * 64,
                tokenizer_hash="b" * 64,
                checkpoint_hash="c" * 64,
                template_hash="d" * 64,
                max_length=1_024,
                objective="dpo",
            )


if __name__ == "__main__":
    unittest.main()
