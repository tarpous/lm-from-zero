# Session handoff: dense vertical-slice implementation

## Current update (July 26, 2026)

The Safetensors checkpoint objective described below is complete and the older
instructions are retained only as historical context. Work has advanced through:

- atomic checkpoint/recovery with exact CPU resume and retention;
- the dry-run-first single-process and `torchrun` DDP dense trainer with
  reduced global metrics, rank-zero logs/checkpoints, and rank-local
  cursor/RNG recovery;
- canonical durable JSONL metrics, resume-aware rank-zero TensorBoard scalars,
  and atomic typed Parquet snapshots/rebuilding;
- fixed non-repeating checkpoint loss evaluation;
- standard local `Olmo2ForCausalLM` export with tokenizer/config metadata,
  explicit tensor mapping, checksums, atomic publication, and fp32/cache parity;
- native dense greedy and seeded temperature/top-k/top-p generation with
  batched prompts, EOS/context handling, KV caching, default control-token
  suppression, and token-step streaming.

The pretraining CLI exposes `--stop-after-optimizer-step` so a smoke can retain
the full scheduler/configuration hash, stop after a few steps, and resume to a
later bound without pretending a tiny-budget scheduler is equivalent.

The dependency boundary now pins `safetensors==0.8.0`,
`transformers==5.14.1`, and the compatible stable `tokenizers==0.22.2`. The
existing ignored TinyStories sample, 16K tokenizer, 23.8M-token shard build,
tokenizer/model binding, and 20.159M dense configuration were revalidated
successfully. WSL sees the RTX 4080 SUPER, and the full 500M-token command was
checked in dry-run mode. The complete offline gate passes Ruff formatting and
lint, strict mypy, CLI/torchrun discovery, lock validation, and 127 tests with
85.56% branch coverage.

The measured dense integration slice is complete. A compiled bf16 CUDA run
used the full-run scheduler configuration, stopped at optimizer step 2, resumed
from that checkpoint through step 4, evaluated one fixed validation batch,
exported a standard local OLMo2 artifact with exact fp32 parity, and completed
cached greedy generation. The portable generated evidence, including
throughput, peak CUDA memory, lineage, source and artifact hashes, lives at
[`reports/zero-20m-tinystories-smoke.json`](reports/zero-20m-tinystories-smoke.json).
This is integration evidence only; four optimizer steps do not establish model
quality.

The next plan work is the remaining Milestone 4 training-system scope: measured
compiled-CUDA resume tolerance. The 500M-token baseline is still a long GPU job
and requires a fresh explicit approval. Data downloads, publication, and other
external changes retain their own approval gates.

---

## Historical checkpoint handoff

## Resume objective

Continue the dense vertical-slice implementation at the atomic Safetensors
checkpoint milestone in section 7.4 of `../plans/01-lm-from-zero.md`.

Do not start a training run yet. The checkpoint and recovery contract must be
implemented and tested before a runnable pretraining command is added.

## Read before changing anything

1. Read the repository-root `AGENTS.md` and follow its approval gates.
2. Read sections 2.1, 6, 7.1, 7.3, 7.4, 15.6, and 17 of
   `../plans/01-lm-from-zero.md`.
3. Inspect `git status`, `pyproject.toml`, `uv.lock`, and the existing training
   APIs before editing.

The plan currently lives one directory above this repository and is therefore
not tracked here.

## Baseline state

- Baseline source commit: `03a2006d9577442e12acff24b6e87db7cdb8ccbe`.
- `main` tracked `origin/main`, and the worktree was clean before this handoff
  file was created.
- Primary environment: WSL2 distro `Ubuntu-24.04`, Python 3.12, project-local
  `.venv`, and `uv`.
- PyTorch 2.13.0 is pinned to the CUDA 13.0 index in `pyproject.toml`.
- Safetensors is not yet declared in `pyproject.toml`, present in `uv.lock`, or
  installed as an approved project dependency.
- The last full gate before the baseline commit passed Ruff formatting, Ruff
  lint, strict mypy, 86 tests, the 85% coverage gate (87.29% measured), lock
  validation, and the CLI help smoke test.

## What is already implemented

- Deterministic TinyStories sampling, the 16K project-owned byte BPE tokenizer,
  checked `uint16` token shards, and rank-aware exact data cursors.
- The validated 20.159M-parameter OLMo2-compatible dense model, including RoPE,
  GQA, QK normalization, SwiGLU, branch post-normalization, shifted causal loss,
  and KV-cache behavior.
- `lm_from_zero.training.ShardBatchSource`, `CausalBatchConfig`, and
  `BatchCursor` for deterministic memory-mapped batches and exact cursor resume.
- Auditable AdamW decay/no-decay partitioning, the pinned warmup/cosine learning
  rate policy, and global gradient clipping in
  `src/lm_from_zero/training/optimization.py`.
- In-memory optimizer-state round-trip coverage that reproduces the next CPU
  step. There is no on-disk training checkpoint implementation yet.

The README's opening status text still says the dense model is next. Update it
as part of this milestone so it accurately reflects the implemented model and
training foundations.

## First approval gate: install Safetensors

Dependency resolution and installation change `pyproject.toml`, `uv.lock`, and
the project-local `.venv`, and may use the network/package cache. Before doing
that, tell the user:

- the exact Safetensors version proposed after checking Python 3.12 and pinned
  PyTorch compatibility;
- that it will be a pinned direct runtime dependency;
- that only this repository's dependency files and `.venv` will change; and
- that it is required to store model weights separately from recovery state.

Then ask for explicit confirmation. After approval, use `uv` from
`Ubuntu-24.04`, preserve the complete lock, and use copy link mode. Do not
install globally or alter the host Python/CUDA setup. Verify the import and
frozen synchronization after the change.

## Checkpoint milestone to implement

Prefer a focused module such as
`src/lm_from_zero/training/checkpointing.py`, exported through
`src/lm_from_zero/training/__init__.py`, with dedicated tests. Keep checkpoint
artifacts under ignored directories.

The implementation must satisfy the plan's recovery contract:

- Publish a checkpoint atomically so an interrupted or partial directory is
  never accepted as complete.
- Store model tensors in a separate Safetensors file.
- Store optimizer, scheduler/training progress, and optional scaler state
  separately from model weights.
- Capture Python, NumPy, Torch CPU, and available CUDA RNG states.
- Persist the exact `BatchCursor`, optimizer step, token counters, best metric,
  and checkpoint lineage.
- Bind recovery to the resolved model configuration, tokenizer hash, shard
  manifest/hash, architecture, rank/world size, dependency versions, hardware
  metadata, and Git revision/dirty state.
- Validate the full manifest before mutating a live model or optimizer.
- Refuse missing, corrupt, mismatched, incomplete, or unsupported checkpoint
  formats with clear errors.
- Resume CPU training bit-exactly. CUDA/compiled resume may claim only measured
  tolerance equivalence.
- Support the retention policy of the latest three checkpoints plus the best
  validation checkpoint without deleting the only valid recovery point.
- Represent both the 250-step and 15-minute save triggers without writing the
  same optimizer step twice. Integration with the eventual runner may be a
  follow-up if storage/recovery remains cleanly separated.

Use canonical, versioned manifests, hashes, atomic writes, strict Pydantic
validation, and negative-path tests consistently with the existing tokenizer
and shard code.

## Minimum acceptance tests

- Safetensors model-state save/load round trip, including dtype and tensor-name
  preservation.
- Full checkpoint round trip that reproduces the exact next CPU batch, loss,
  parameter update, optimizer state, learning rate, counters, and RNG draws.
- Save interruption leaves the previous complete checkpoint usable and the
  partial replacement unaccepted.
- Missing/corrupt weight, state, or manifest files fail before restore.
- Model configuration, tokenizer, shard, architecture, rank/world-size, and
  format-version mismatches are rejected.
- Retention keeps the latest three plus a distinct best checkpoint.
- Duplicate step/time triggers do not create duplicate checkpoints.

Avoid pickle-loading untrusted artifacts where a safer restricted loading path
is available. Document the trust boundary for optimizer/recovery state.

## Verification before handing back

Run from WSL2 after the approved dependency synchronization:

```bash
PYTHONPATH=src uv run --frozen ruff format --check .
PYTHONPATH=src uv run --frozen ruff check .
PYTHONPATH=src uv run --frozen mypy src tests
PYTHONPATH=src uv run --frozen pytest
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli --help
uv lock --check
```

Also run narrow checkpoint tests during development. Update `README.md` and
`AGENTS.md` if setup, commands, checkpoint layout, or verification behavior
changes.

Do not run a long GPU job, download data, publish artifacts, commit, or push
without a fresh explicit user confirmation for that action.
