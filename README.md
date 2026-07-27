# lm-from-zero

`lm-from-zero` is a from-scratch language-model engineering project covering a
byte-level BPE tokenizer, deterministic data preparation, three model families,
resumable training, post-training, export, local inference, and controlled
experiments.

The implementation follows `../plans/01-lm-from-zero.md`. Work proceeds in
milestone order, beginning with this runnable dense vertical slice:

> TinyStories -> byte BPE -> checked token shards -> 20M dense model ->
> resumable checkpoint -> evaluation -> Hugging Face export -> local inference

The repository has completed the checked TinyStories sample, pedagogical 16K
tokenizer, deterministic token shards, dense 20M model, deterministic batch and
optimizer foundations, atomic resumable checkpoints, the single-process and
DDP dense pretraining runner, deterministic checkpoint loss evaluation, local
Hugging Face OLMo2 export with parity validation, and native cached causal
generation. A compiled bf16 CUDA smoke on the RTX 4080 SUPER has exercised
training, checkpoint resume, evaluation, export, and native generation. Its
generated evidence is committed at
[`reports/zero-20m-tinystories-smoke.json`](reports/zero-20m-tinystories-smoke.json).
No dataset, model weight, or external service is needed for the offline test
suite.

## Environment

The primary target is the `Ubuntu-24.04` WSL2 distro with Python 3.12 and `uv`.
The machine also has a distro named `Ubuntu`, but `uv` is not available there.
From PowerShell, enter the intended distro with:

```powershell
wsl -d Ubuntu-24.04
```

CPU development and offline tests must also remain usable on Windows once a
Windows Python 3.12 runtime is available.

The approved initial toolchain uses a project-local environment:

```bash
uv venv --python 3.12 .venv
uv sync --frozen --all-groups --link-mode copy
```

PyTorch is pinned to the official CUDA 13.0 wheel index through an explicit uv
source. The index can provide only `torch`; all unrelated dependencies continue
to resolve from PyPI. CUDA libraries and PyTorch remain inside `.venv` and the
existing uv cache rather than modifying the host CUDA toolkit.

Transformers 5.14.1 is used only at the export/comparison boundary. Its declared
compatibility range requires Tokenizers 0.22.2; core tokenizer training, model
definitions, pretraining, and objectives do not import Transformers.
TensorBoard 2.21.0 supplies the local `torch.utils.tensorboard.SummaryWriter`
used by rank-zero training telemetry. Polars and PyArrow materialize the same
canonical metrics as a typed Parquet table.

Do not install project packages globally. Optional vLLM, JAX-CUDA, Mamba oracle,
llama.cpp, GPU, dataset, and publication workflows will be documented and kept
separate from the default environment.

## Verification

Run the complete offline CPU quality gate from WSL:

```bash
PYTHONPATH=src uv run --frozen ruff format --check .
PYTHONPATH=src uv run --frozen ruff check .
PYTHONPATH=src uv run --frozen mypy src tests
PYTHONPATH=src uv run --frozen pytest
```

Pytest enforces branch coverage of at least 85%.

## Dependency-free fallback

The standard-library suite remains available when dependencies are not synced:

```bash
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
python3.12 -m compileall -q src tests
```

Or on Windows:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m unittest discover -s tests -v
py -3.12 -m compileall -q src tests
```

The default verification path performs no network access.

## Implemented

- Fixed special-token IDs 0-7.
- A deterministic, pure-Python byte-level BPE trainer.
- Byte vocabulary ordering and merge tie-breaks matching the Hugging Face
  ByteLevel BPE oracle under identical settings.
- Byte-exact encoding and decoding with no unknown token.
- Opt-in special-token parsing so ordinary text cannot silently become control
  tokens.
- Canonical, versioned tokenizer serialization and SHA-256 model hashes.
- Hypothesis fuzz tests and unit tests for arbitrary bytes, Unicode,
  deterministic merge selection, oracle parity, special-token isolation,
  validation, and save/load parity.
- Stable content-hash train/validation/test assignment with duplicate rejection.
- Atomic little-endian `uint16` token shards with EOS document boundaries,
  SHA-256 manifests, tokenizer/source provenance, and resumable document cursors.
- Read-only memory-mapped shard loading with length, checksum, vocabulary,
  orphan, partial-file, mixed-tokenizer, and cross-shard duplicate checks.
- Serializable rank-aware stride cursors with no overlap between ranks.
- A deterministic, size-bounded TinyStories streamer pinned to an immutable Git
  revision, with exact deduplication counts and per-document, aggregate, and
  artifact hashes.
- Incremental indexed BPE training with deterministic replay from atomic
  checkpoints; full-corpus pair recounts are not required after each merge.
- Exact full-merge and encoding-ID parity against the Hugging Face Tokenizers
  Rust oracle.
- Append-only training, oracle, throughput, and traced-memory benchmark records.
- A validated OLMo2-compatible dense configuration with canonical config hashes
  and analytic parameter/FLOP breakdowns.
- Project-owned RoPE, grouped-query SDPA, flat QK RMSNorm, bias-free SwiGLU,
  exact post-branch normalization, untied embeddings/output head, shifted causal
  loss, and dynamic grouped KV caches.
- CPU behavior tests for causal isolation, padding, cache parity, loss,
  gradients, runtime validation, and the authoritative 20M parameter count.
- Validated memory-mapped training windows with stable hash shuffling,
  disjoint rank partitioning, immutable exact-resume cursors, and deterministic
  epoch rollover.
- Auditable AdamW decay/no-decay groups, global gradient clipping, and the
  pinned 1.5% linear-warmup/cosine-decay learning-rate policy.
- Atomic, versioned training checkpoints with separate Safetensors model
  weights, restricted-load recovery state, full artifact hashes, exact data
  cursors and RNG state, environment/source bindings, lineage, retention, and
  duplicate-safe step/time triggers.
- A dry-run-first dense pretraining command with fixed global effective
  batches, gradient accumulation, clipping and scheduling, single-process and
  `torchrun` DDP execution, reduced metrics, rank-zero JSONL/checkpoint
  ownership, rank-local recovery state, bounded CPU smoke support, checkpoint
  resume, and final checkpoint publication.
- Durable canonical training JSONL, resume-aware rank-zero TensorBoard
  scalars, and atomic typed Parquet metric snapshots with deterministic
  last-record-per-step recovery.
- Fixed non-repeating shard evaluation with validated model-only checkpoint
  loading, causal loss, perplexity, throughput, exact cursors, and canonical
  append-only JSONL results.
- Standard local `Olmo2ForCausalLM` export with an explicit complete tensor map,
  standard tokenizer and generation metadata, internal/HF fp32 and KV-cache
  parity, atomic publication, and a canonical checksummed provenance manifest.
- Native dense generation with greedy, temperature, top-k, and top-p decoding;
  deterministic seeds; variable-length batches; EOS/context handling; default
  control-token suppression; dense KV reuse; and token-step streaming events.

## Dense 20M model

The pinned `zero-20m-tinystories` configuration uses five layers, width 384,
six query heads, two key/value heads, FFN width 1,024, vocabulary 16,000, and a
1,024-token context. It targets `Olmo2ForCausalLM` export semantics: Q/K and
branch-output RMSNorm, no pre-norm, and no tied weights.

Validate the exact tokenizer binding, instantiate the model, compare realized
and analytic parameter counts, and print the canonical configuration summary:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  dense-model-summary artifacts/tokenizers/tinystories-16k/training.json
```

The command derives every reported value from the checked tokenizer manifest
and current implementation; measured values are not copied into this README.

Training data, optimizer, checkpoint, and single-process runner components are
typed Python APIs under `lm_from_zero.training`. The four-step run documented
in the generated smoke report is integration evidence, not a trained model or
quality result. The 500M-token baseline remains an explicit long-run approval
boundary.

## Training checkpoints

Each published recovery point is an immutable directory under an ignored
artifact root:

```text
artifacts/checkpoints/<run>/step-000000000250/
├── manifest.json
├── model.safetensors
└── recovery.pt
```

`manifest.json` is canonical, versioned JSON. It hashes both payload files and
binds recovery to the resolved model configuration, architecture, tokenizer,
shard manifest, rank/world size, dependency versions, hardware facts, Git
revision/dirty state, optimizer and scheduler progress, exact `BatchCursor`,
best metric, and parent checkpoint.

`model.safetensors` contains only named model tensors. `recovery.pt` contains
optimizer, scheduler, optional scaler, and Python/NumPy/Torch CPU/CUDA RNG
state. Recovery verifies the complete manifest, hashes, binding, model tensor
names/shapes/dtypes, optimizer layout, and RNG structure before mutating live
training objects. The recovery payload is loaded with
`torch.load(weights_only=True)`; despite that restricted loader, accept
checkpoints only from a trusted training run whose manifest hashes validate.

Saving writes and validates a hidden sibling directory before one atomic rename.
An interrupted save cannot replace a prior recovery point. Cadence supports
both every 250 optimizer steps and every 15 minutes without writing one step
twice. Retention keeps the latest three valid checkpoints plus a distinct valid
best checkpoint and never removes the sole valid recovery point.

Run the focused checkpoint tests with:

```bash
PYTHONPATH=src uv run --frozen pytest tests/test_checkpointing.py --no-cov
```

## Dense pretraining

`pretrain-dense` validates the complete shard build and prints an auditable
dry-run plan by default. The plan includes the resolved configuration hashes,
effective tokens per optimizer step, rounded token budget, analytic training
FLOPs, estimated checkpoint size, device/precision/compile policy, and optional
wall-time estimate:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli pretrain-dense \
  artifacts/shards/tinystories-16k/build.json \
  --checkpoint-directory artifacts/checkpoints/zero-20m-tinystories \
  --jsonl-log artifacts/runs/zero-20m-tinystories/events.jsonl \
  --tensorboard-directory artifacts/runs/zero-20m-tinystories/tensorboard \
  --parquet-log artifacts/runs/zero-20m-tinystories/metrics.parquet \
  --target-tokens 500000000 \
  --estimated-tokens-per-second <measured-throughput>
```

This command does not allocate the model or train unless `--execute` is added.
Review the dry-run output and obtain explicit approval before using
`--execute`, even for the full local GPU run. A short bounded CPU/GPU smoke must
precede that long run.

For a resumable smoke, keep the real target-token configuration and add
`--stop-after-optimizer-step <step>` with `--execute`. Resume from that
checkpoint with the same configuration hash and a later bounded step. Do not
replace this with a tiny total token budget: that would test a different
scheduler and make the checkpoint intentionally incompatible with the real run.

The runner uses the pinned AdamW and warmup/cosine policy, divides each
microbatch loss by the accumulation count, clips the resulting global gradient,
and writes step/checkpoint/resume/completion events to canonical append-only
JSONL. Every optimizer-step event includes measured elapsed time, effective
token throughput, and CUDA peak allocated/reserved bytes when applicable. It
saves a final checkpoint even when neither periodic trigger fires. Resume
rejects a changed training-configuration hash as well as the checkpoint binding
mismatches documented above.

When the two metric options are omitted, TensorBoard defaults to a
`tensorboard/` sibling of the JSONL log and Parquet defaults to the JSONL path
with a `.parquet` suffix. JSONL is appended and fsynced first, making it the
durable recovery source. TensorBoard uses `purge_step` on resume so steps after
the restored checkpoint are replaced rather than displayed twice. Parquet is
atomically rebuilt at checkpoint snapshots and graceful termination, keeping
the latest canonical record for each optimizer step.

After an abrupt interruption, rebuild Parquet directly from the surviving
canonical JSONL:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  materialize-training-metrics \
  artifacts/runs/zero-20m-tinystories/events.jsonl \
  --output artifacts/runs/zero-20m-tinystories/metrics.parquet
```

The materializer rejects malformed, noncanonical, or schema-invalid
optimizer-step records and publishes through an atomic replacement. All metric
artifacts remain under ignored local artifact directories.

Launch DDP through `torchrun`; each process receives a disjoint deterministic
rank shard, while the token budget and reported throughput remain global:

```bash
PYTHONPATH=src uv run --frozen torchrun --standalone --nproc-per-node=2 \
  -m lm_from_zero.cli pretrain-dense \
  artifacts/shards/tinystories-16k/build.json \
  --checkpoint-directory artifacts/checkpoints/ddp-smoke \
  --jsonl-log artifacts/runs/ddp-smoke/events.jsonl \
  --target-tokens 32768 \
  --sequence-length 1024 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --device cpu \
  --no-compile-model \
  --stop-after-optimizer-step 2 \
  --execute
```

Only rank zero writes JSONL or publishes and prunes checkpoints. Checkpoint
timing decisions are broadcast before synchronization, and the restricted-load
recovery payload stores every rank's exact cursor and safe RNG state. Resume
restores the common model/optimizer state and then the matching rank-local
cursor/RNG. Configuration hashes normalize away the process-local rank while
retaining world size, so a checkpoint cannot resume under a different process
count. Object collectives exchange only trusted same-run control data and
plain integer RNG representations; checkpoint loading remains restricted to
`torch.load(weights_only=True)`.

Focused offline runner verification:

```bash
PYTHONPATH=src uv run --frozen pytest tests/test_runner.py --no-cov
```

The runner suite includes a real two-process CPU/Gloo interruption/resume test
and compares its final parameters bit-exactly with an uninterrupted DDP run. A
multi-GPU test remains optional and requires separate hosted or additional
hardware approval. Compiled-CUDA resume-tolerance measurement remains the
final Milestone 4 follow-up.

## Dense checkpoint evaluation

Evaluate a complete checkpoint on fixed validation batches without restoring
optimizer or RNG state:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli evaluate-dense \
  artifacts/checkpoints/zero-20m-tinystories/step-000000000100 \
  artifacts/shards/tinystories-16k/build.json \
  --split validation \
  --sequence-length 1024 \
  --batch-size 8 \
  --max-batches 32 \
  --device cuda \
  --precision bf16 \
  --jsonl-output artifacts/evaluations/zero-20m-tinystories.jsonl
```

Evaluation validates all checkpoint payload hashes and bindings before loading
the model. It refuses requests that would wrap into another shard epoch and
therefore repeat validation windows. The result records causal loss,
perplexity, predicted-token throughput, exact start/end cursors, model/tokenizer
hashes, and the shard-build hash. Bits-per-byte and downstream task evaluation
remain later evaluation-harness work.

Focused offline evaluation verification:

```bash
PYTHONPATH=src uv run --frozen pytest tests/test_evaluation.py --no-cov
```

## Dense Hugging Face export

Export a validated dense checkpoint and its exact tokenizer as a standard local
`Olmo2ForCausalLM` directory:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli export-dense-hf \
  artifacts/checkpoints/zero-20m-tinystories/step-000000000100 \
  artifacts/tokenizers/tinystories-16k/training.json \
  --output-directory artifacts/exports/zero-20m-tinystories
```

The command validates the complete source checkpoint and tokenizer lineage
before loading weights. It maps every tensor by an explicit name, rejects
unknown, missing, duplicate, or shape-incompatible tensors, and requires
internal/Transformers fp32 logits to match with `atol=1e-5, rtol=1e-5`.
It then writes `model.safetensors`, standard model/tokenizer/generation
configuration, tokenizer metadata, and `export_manifest.json` into a hidden
sibling directory before one atomic rename. The manifest records source hashes,
the exact tensor map, measured parity error, and SHA-256/size metadata for every
exported file.

The output directory must not already exist. This workflow is local and offline:
it does not access the Hugging Face Hub, authenticate, create a repository, or
publish anything. Hub publication remains a separate approval gate.

Focused offline export, reload, tokenizer, parity, corruption, and interruption
verification:

```bash
PYTHONPATH=src uv run --frozen pytest tests/test_export_hf.py --no-cov
```

## Native dense generation

Generate from a validated local checkpoint with the project-owned model and KV
cache:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli generate-dense \
  artifacts/checkpoints/zero-20m-tinystories/step-000000000100 \
  artifacts/tokenizers/tinystories-16k/training.json \
  "Once upon a time" \
  --max-new-tokens 128 \
  --strategy sample \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95 \
  --seed 1337 \
  --device cuda \
  --stream \
  --jsonl-output artifacts/generations/zero-20m-tinystories.jsonl
```

Omit the sampling options and keep `--strategy greedy` for deterministic greedy
decoding. `--stream` emits one JSON token event per model step followed by the
complete result. The typed Python API also accepts variable-length prompt
batches and stops each sequence independently at EOS. `pad`, role, and `mask`
IDs are suppressed by default; `--allow-raw-special-tokens` is intended only
for explicit diagnostics. `--jsonl-output` appends a canonical, fsynced record
containing measured generation results plus model, tokenizer, and prompt-token
hashes; prompt text itself is not stored.

The command validates every checkpoint payload and the exact tokenizer binding
before loading the model. It rejects empty prompts, out-of-vocabulary tokens,
invalid sampling combinations, and requests exceeding the configured context.
It reports actual model forwards, elapsed time, generated tokens, and measured
tokens per second. No network or hosted runtime is used.

Focused native generation verification:

```bash
PYTHONPATH=src uv run --frozen pytest tests/test_generation.py --no-cov
```

Generate a compact, portable smoke report only after training, resume,
evaluation, export, and generation records exist:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  build-dense-smoke-report \
  artifacts/runs/zero-20m-tinystories-smoke/events.jsonl \
  artifacts/checkpoints/zero-20m-tinystories-smoke/step-000000000004 \
  artifacts/evaluations/zero-20m-tinystories-smoke.jsonl \
  artifacts/exports/zero-20m-tinystories-smoke \
  artifacts/generations/zero-20m-tinystories-smoke.jsonl \
  --output reports/zero-20m-tinystories-smoke.json
```

The builder validates checkpoint contents, clean Git lineage, contiguous
optimizer steps, resume ancestry, and matching model/tokenizer/shard/checkpoint
hashes across every input. The committed report is generated from recorded
artifacts; measured values are not copied manually. It records the resumed
four-step compiled bf16 CUDA run, evaluation throughput, exact Hugging Face
fp32 parity, native cached-generation throughput, artifact hashes, and the
source revision. The intentionally untrained generation output is not treated
as a quality result.

## TinyStories tokenizer sample

The approved tokenizer sample uses public dataset revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, selects a deduplicated prefix in
source order, and stops after at least 100,000,000 UTF-8 text bytes. It requires
no Hugging Face login.

```bash
HF_HOME=.cache/huggingface \
HF_DATASETS_CACHE=.cache/huggingface/datasets \
HF_HUB_DISABLE_TELEMETRY=1 \
DO_NOT_TRACK=1 \
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  sample-tinystories \
  --output-directory data/tinystories \
  --cache-directory .cache/huggingface \
  --target-text-bytes 100000000 \
  --max-storage-bytes 1000000000
```

Both directories are ignored by Git. Verify every document and recorded hash:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli verify-sample \
  data/tinystories/manifest.json
```

## Train and verify the 16K tokenizer

Train or resume the required project-owned tokenizer:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli train-tokenizer \
  data/tinystories/manifest.json \
  --output-directory artifacts/tokenizers/tinystories-16k \
  --target-vocab-size 16000 \
  --min-frequency 2 \
  --checkpoint-every-merges 250 \
  --max-corpus-bytes 100000000
```

Retrain the Rust oracle under identical settings and require exact parity:

```bash
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  verify-tokenizer-oracle \
  artifacts/tokenizers/tinystories-16k/training.json \
  data/tinystories/manifest.json \
  --jsonl-output artifacts/benchmarks/tokenizer_oracle.jsonl
```

Measure throughput separately from traced Python allocations:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  benchmark-tokenizer \
  artifacts/tokenizers/tinystories-16k/tokenizer.json \
  data/tinystories/manifest.json \
  --max-text-bytes 10000000 \
  --jsonl-output artifacts/benchmarks/tokenizer_encode.jsonl
```

Add `--trace-memory` for the allocation-measurement run. Tracing adds enough
overhead that its throughput is not used as the uninstrumented speed result.

## Token shard construction and verification

Build the deterministic 98/1/1 content-hash splits and atomically publish
checked 100M-token-cap `uint16` shards:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  build-token-shards \
  data/tinystories/manifest.json \
  artifacts/tokenizers/tinystories-16k/training.json \
  --output-directory artifacts/shards/tinystories-16k
```

The final directory appears only after the sample, tokenizer lineage, every
shard checksum, token range, cursor, source-document hash, and cross-split
deduplication check succeeds. A failed attempt leaves a sibling
`.tinystories-16k.partial` directory for explicit inspection; it is never
accepted as training data or silently overwritten.

Validate the complete build before training:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  verify-shard-build artifacts/shards/tinystories-16k/build.json
```

Validate a completed shard before training:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli verify-shard \
  artifacts/shards/tinystories-16k/shards/train-00000.json \
  --tokenizer-hash <sha256> \
  --vocab-size 16000
```

Both verification commands are local-only and print checked JSON metadata.

## Repository layout

```text
src/lm_from_zero/       project-owned implementation
tests/                  offline CPU tests
.github/workflows/      locked CPU quality gate
```

Later milestone directories will be added alongside working behavior rather
than as empty stubs.

## Approval gates

Explicit confirmation is required before:

- creating an environment or resolving/installing dependencies;
- downloading datasets, model weights, binaries, or optional oracle packages;
- creating containers, external volumes, or project caches;
- running long GPU or hosted jobs;
- logging into or changing GitHub, Hugging Face, or another external service;
- publishing artifacts, pushing commits, opening pull requests, or creating
  releases.

Before each gate, the proposed command, destination, approximate size or cost,
and purpose must be stated.

## Status

The plan is deliberately larger than a single implementation pass. Measured
throughput, quality, memory, and compatibility claims will be added only after
the corresponding runs exist.
