"""Hugging Face Tokenizers comparison oracle for project-owned BPE."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from lm_from_zero.data import DataValidationError
from lm_from_zero.sampling import load_sample
from lm_from_zero.tokenizer.bpe import (
    BYTE_LEVEL_SYMBOLS,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    ByteBPE,
)
from lm_from_zero.tokenizer.pipeline import limit_texts, load_training_manifest

BYTE_LEVEL_DECODER = {
    symbol: byte_value for byte_value, symbol in enumerate(BYTE_LEVEL_SYMBOLS)
}


class OracleParityResult(BaseModel):
    """Measured full-merge and fixed-prompt parity outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokenizer_hash: str
    sample_sha256: str
    merge_count: Annotated[int, Field(ge=0)]
    prompt_count: Annotated[int, Field(gt=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    merge_order_equal: bool
    encoding_ids_equal: bool


def _decode_symbol(symbol: str) -> bytes:
    return bytes(BYTE_LEVEL_DECODER[character] for character in symbol)


def _oracle_merges(tokenizer: Tokenizer) -> tuple[tuple[bytes, bytes], ...]:
    with tempfile.TemporaryDirectory() as directory:
        tokenizer.model.save(directory)
        lines = (
            (Path(directory) / "merges.txt")
            .read_text(encoding="utf-8")
            .splitlines()[1:]
        )
    result: list[tuple[bytes, bytes]] = []
    for line in lines:
        left, right = line.split()
        result.append((_decode_symbol(left), _decode_symbol(right)))
    return tuple(result)


def verify_hugging_face_oracle(
    training_manifest_path: str | Path,
    sample_manifest_path: str | Path,
) -> OracleParityResult:
    """Retrain the Rust oracle under identical settings and require exact parity."""

    training_path = Path(training_manifest_path)
    training = load_training_manifest(training_path)
    if training.status != "complete":
        raise DataValidationError("oracle parity requires completed tokenizer training")
    tokenizer = ByteBPE.load(training_path.parent / training.tokenizer_file)
    if tokenizer.model_hash != training.tokenizer_hash:
        raise DataValidationError("trained tokenizer hash does not match manifest")
    sample, texts = load_sample(sample_manifest_path)
    if sample.sample_sha256 != training.source_sample_sha256:
        raise DataValidationError("oracle sample does not match training manifest")

    oracle = Tokenizer(models.BPE())
    oracle.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=training.training_config.target_vocab_size,
        min_frequency=training.training_config.min_frequency,
        show_progress=False,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    started = perf_counter()
    oracle.train_from_iterator(
        limit_texts(texts, training.training_config),
        trainer=trainer,
    )

    for token, token_id in SPECIAL_TOKEN_IDS.items():
        if oracle.token_to_id(token) != token_id:
            raise DataValidationError(f"oracle special-token ID mismatch: {token}")
    project_merges = tuple(
        (tokenizer.token_bytes(left), tokenizer.token_bytes(right))
        for left, right in tokenizer.merges
    )
    oracle_merges = _oracle_merges(oracle)
    merge_order_equal = project_merges == oracle_merges
    if not merge_order_equal:
        mismatch = next(
            (
                index
                for index, (project, reference) in enumerate(
                    zip(project_merges, oracle_merges, strict=False)
                )
                if project != reference
            ),
            min(len(project_merges), len(oracle_merges)),
        )
        raise DataValidationError(f"oracle merge-order mismatch at index {mismatch}")

    prompts = (
        "Once upon a time, there was a little fox.",
        "I'm here, and you're there!",
        "Numbers 42 and 4242; naïve café.\n",
    )
    encoding_ids_equal = all(
        tokenizer.encode(prompt) == oracle.encode(prompt).ids for prompt in prompts
    )
    if not encoding_ids_equal:
        raise DataValidationError("oracle encoding ID mismatch")
    return OracleParityResult(
        tokenizer_hash=tokenizer.model_hash,
        sample_sha256=sample.sample_sha256,
        merge_count=len(tokenizer.merges),
        prompt_count=len(prompts),
        elapsed_seconds=perf_counter() - started,
        merge_order_equal=True,
        encoding_ids_equal=True,
    )


def append_oracle_result(path: str | Path, result: OracleParityResult) -> None:
    """Append one canonical oracle record."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    descriptor = os.open(
        destination,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
