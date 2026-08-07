from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from pydantic import ValidationError

from lm_from_zero.post_training import (
    DEFAULT_CHAT_TEMPLATE,
    ChatMessage,
    ChatRole,
    Conversation,
    SFTConfig,
    SFTFormatError,
    assistant_only_causal_loss,
    collate_supervised_chat,
    render_supervised_chat,
)
from lm_from_zero.post_training.sft import IGNORE_INDEX
from lm_from_zero.tokenizer.bpe import SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, ByteBPE


def _conversation(*messages: tuple[ChatRole, str]) -> Conversation:
    return Conversation(
        messages=tuple(
            ChatMessage(role=role, content=content) for role, content in messages
        )
    )


class SFTTests(unittest.TestCase):
    def test_template_is_versioned_and_turns_are_validated(self) -> None:
        conversation = _conversation(
            ("system", "Be concise."),
            ("user", "Hello"),
            ("assistant", "Hi"),
        )
        self.assertEqual(
            DEFAULT_CHAT_TEMPLATE.render(conversation),
            "<|bos|><|system|>Be concise.<|end|>"
            "<|user|>Hello<|end|><|assistant|>Hi<|end|><|eos|>",
        )
        self.assertEqual(len(DEFAULT_CHAT_TEMPLATE.template_hash), 64)
        self.assertEqual(
            tuple(DEFAULT_CHAT_TEMPLATE.role_tokens),
            ("system", "user", "assistant"),
        )
        with self.assertRaisesRegex(ValidationError, "reserved control tokens"):
            ChatMessage(role="user", content="use <|assistant|> here")
        with self.assertRaisesRegex(ValidationError, "must begin"):
            Conversation(messages=(ChatMessage(role="assistant", content="no"),))
        with self.assertRaisesRegex(ValidationError, "expected a user"):
            Conversation(
                messages=(
                    ChatMessage(role="system", content="rules"),
                    ChatMessage(role="assistant", content="wrong"),
                )
            )
        with self.assertRaisesRegex(ValidationError, "expected a assistant"):
            Conversation(
                messages=(
                    ChatMessage(role="user", content="one"),
                    ChatMessage(role="user", content="two"),
                )
            )

    def test_supervised_render_masks_prompt_and_keeps_empty_turn_boundaries(
        self,
    ) -> None:
        tokenizer = ByteBPE()
        conversation = _conversation(
            ("system", "rules"),
            ("user", "question"),
            ("assistant", "answer"),
            ("user", "follow-up"),
            ("assistant", ""),
        )
        example = render_supervised_chat(conversation, tokenizer)
        expected = tuple(
            tokenizer.encode(
                DEFAULT_CHAT_TEMPLATE.render(conversation),
                allowed_special=SPECIAL_TOKENS,
            )
        )
        self.assertEqual(example.input_ids, expected)
        self.assertEqual(example.input_ids[0], SPECIAL_TOKEN_IDS["<|bos|>"])
        self.assertEqual(example.input_ids[-1], SPECIAL_TOKEN_IDS["<|eos|>"])
        self.assertFalse(example.truncated)
        self.assertGreater(example.assistant_token_count, 0)
        self.assertEqual(
            example.labels.count(IGNORE_INDEX) + example.assistant_token_count,
            len(example.labels),
        )
        first_assistant = example.input_ids.index(SPECIAL_TOKEN_IDS["<|assistant|>"])
        first_end = example.input_ids.index(
            SPECIAL_TOKEN_IDS["<|end|>"], first_assistant
        )
        self.assertTrue(
            all(
                label != IGNORE_INDEX
                for label in example.labels[first_assistant + 1 : first_end + 1]
            )
        )
        self.assertEqual(example.labels[first_assistant], IGNORE_INDEX)
        with self.assertRaisesRegex(SFTFormatError, "end with an assistant"):
            render_supervised_chat(_conversation(("user", "prompt")), tokenizer)
        with self.assertRaisesRegex(SFTFormatError, "end with an assistant"):
            render_supervised_chat(_conversation(("system", "rules")), tokenizer)

    def test_left_truncation_drops_oldest_pairs_before_content(self) -> None:
        tokenizer = ByteBPE()
        conversation = _conversation(
            ("system", "system context"),
            ("user", "old user message " * 4),
            ("assistant", "old assistant message " * 4),
            ("user", "current user message " * 4),
            ("assistant", "final answer"),
        )
        with self.assertRaisesRegex(SFTFormatError, "exceeds"):
            render_supervised_chat(conversation, tokenizer, max_length=20)
        example = render_supervised_chat(
            conversation,
            tokenizer,
            max_length=20,
            truncation="left",
        )
        self.assertTrue(example.truncated)
        self.assertLessEqual(len(example.input_ids), 20)
        self.assertEqual(example.input_ids[0], SPECIAL_TOKEN_IDS["<|bos|>"])
        self.assertEqual(example.input_ids[-1], SPECIAL_TOKEN_IDS["<|eos|>"])
        self.assertGreater(example.assistant_token_count, 0)
        with self.assertRaisesRegex(SFTFormatError, "greater than one"):
            render_supervised_chat(conversation, tokenizer, max_length=1)
        with self.assertRaisesRegex(SFTFormatError, "unsupported"):
            render_supervised_chat(
                conversation,
                tokenizer,
                truncation="middle",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(SFTFormatError, "control markers"):
            render_supervised_chat(
                _conversation(("user", "u"), ("assistant", "a")),
                tokenizer,
                max_length=5,
                truncation="left",
            )

    def test_sft_batch_collation_and_config(self) -> None:
        tokenizer = ByteBPE()
        examples = (
            render_supervised_chat(
                _conversation(("user", "short"), ("assistant", "answer")),
                tokenizer,
            ),
            render_supervised_chat(
                _conversation(("user", "long prompt"), ("assistant", "long answer")),
                tokenizer,
            ),
        )
        batch = collate_supervised_chat(examples, pad_to_length=40)
        self.assertEqual(batch.input_ids.shape, (2, 40))
        self.assertEqual(batch.labels.shape, (2, 40))
        self.assertEqual(batch.attention_mask.shape, (2, 40))
        self.assertEqual(
            batch.lengths, tuple(len(example.input_ids) for example in examples)
        )
        self.assertTrue(torch.all(batch.attention_mask[0, : batch.lengths[0]]))
        self.assertTrue(torch.all(~batch.attention_mask[0, batch.lengths[0] :]))
        self.assertTrue(torch.all(batch.input_ids[0, batch.lengths[0] :] == 0))
        self.assertTrue(torch.all(batch.labels[0, batch.lengths[0] :] == IGNORE_INDEX))
        self.assertEqual(
            batch.assistant_token_counts,
            tuple(item.assistant_token_count for item in examples),
        )
        self.assertEqual(SFTConfig().max_length, 1_024)
        self.assertEqual(SFTConfig(packing="padding_free").packing, "padding_free")
        with self.assertRaisesRegex(SFTFormatError, "empty SFT batch"):
            collate_supervised_chat(())
        with self.assertRaisesRegex(SFTFormatError, "negative"):
            collate_supervised_chat(examples, pad_token_id=-1)
        with self.assertRaisesRegex(SFTFormatError, "shorter"):
            collate_supervised_chat(examples, pad_to_length=1)

    def test_assistant_only_loss_matches_shifted_cross_entropy(self) -> None:
        torch.manual_seed(1337)
        logits = torch.randn(2, 5, 7, requires_grad=True)
        labels = torch.tensor(
            [
                [IGNORE_INDEX, IGNORE_INDEX, 2, 3, 4],
                [IGNORE_INDEX, 1, IGNORE_INDEX, 5, 6],
            ],
            dtype=torch.long,
        )
        actual = assistant_only_causal_loss(logits, labels)
        expected = F.cross_entropy(
            logits[:, :-1].contiguous().float().view(-1, 7),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=IGNORE_INDEX,
        )
        self.assertTrue(torch.allclose(actual, expected))
        torch.autograd.backward(actual)
        self.assertIsNotNone(logits.grad)
        with self.assertRaisesRegex(SFTFormatError, "shape"):
            assistant_only_causal_loss(torch.randn(5, 7), labels)
        with self.assertRaisesRegex(SFTFormatError, "match"):
            assistant_only_causal_loss(torch.randn(2, 4, 7), labels)
        with self.assertRaisesRegex(SFTFormatError, "torch.long"):
            assistant_only_causal_loss(logits.detach(), labels.float())
        with self.assertRaisesRegex(SFTFormatError, "no assistant"):
            assistant_only_causal_loss(
                torch.randn(1, 3, 7),
                torch.full((1, 3), IGNORE_INDEX, dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()
