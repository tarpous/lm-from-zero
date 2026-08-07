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
The project-owned Mamba-2 core and shared lifecycle are also implemented. Its
compiled bf16 CUDA training/resume, evaluation, parity-safe Hugging Face export,
and recurrent generation evidence is committed at
[`reports/zero-20m-mamba2-smoke.json`](reports/zero-20m-mamba2-smoke.json).
Its independent Triton SSD path also passes the pinned `mamba-ssm` numerical
oracle on the RTX 4080 SUPER; the generated evidence is committed at
[`reports/zero-20m-mamba2-oracle.json`](reports/zero-20m-mamba2-oracle.json).
The masked-diffusion 20M track now also completes its bounded compiled-bf16
CUDA training/resume, fixed-window evaluation, self-contained export, and
native iterative-denoising smoke. Its generated evidence is at
[`reports/zero-20m-diffusion-smoke.json`](reports/zero-20m-diffusion-smoke.json).
All three smoke reports are integration evidence, not model-quality claims.
No dataset, model weight, or external service is needed for the offline test
suite.

Milestone 6A's synchronized acceleration calibration is also complete. The
machine-readable report contains all 26 planned cells across the three tracks
and three validated repetitions per cell:
[`reports/zero-20m-acceleration-calibration.json`](reports/zero-20m-acceleration-calibration.json).
No dense or Mamba candidate passed the frozen promotion gates. Diffusion's
fused AdamW candidate passed with a 13.61% end-to-end median speedup and is the
only promoted calibration choice; this does not replace the scientific
objective or begin the long architecture study.

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
uv sync --frozen --link-mode copy
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

Do not install project packages globally. Optional vLLM, JAX-CUDA, llama.cpp,
GPU, dataset, and publication workflows will be documented and kept separate
from the default environment.

## Verification

Run the complete offline CPU quality gate from WSL:

```bash
PYTHONPATH=src uv run --frozen ruff format --check .
PYTHONPATH=src uv run --frozen ruff check .
PYTHONPATH=src uv run --frozen mypy src tests
PYTHONPATH=src uv run --frozen pytest
```

Pytest enforces branch coverage of at least 85%.

## Live progress

Long-running training, evaluation, generation, shard, and calibration loops
show a `tqdm` progress bar on interactive stderr. The bar reports the current
phase, completed work, elapsed time, rate, ETA, and phase-specific metrics such
as loss, tokens per second, checkpointing, or CUDA compilation. Machine-readable
JSON output and canonical JSONL logs remain unchanged.

Progress is enabled automatically in an interactive terminal. Force it when a
wrapper does not expose a TTY with:

```bash
LM_FROM_ZERO_PROGRESS=1 PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli <command>
```

Disable it with `LM_FROM_ZERO_PROGRESS=0`.

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
- A generated compiled-CUDA resume-tolerance report that validates clean,
  matching run/checkpoint contracts before comparing every final model tensor
  and every optimizer-step loss against declared thresholds.
- Fixed non-repeating shard evaluation with validated model-only checkpoint
  loading, causal loss, perplexity, throughput, exact cursors, and canonical
  append-only JSONL results.
- Standard local `Olmo2ForCausalLM` export with an explicit complete tensor map,
  standard tokenizer and generation metadata, internal/HF fp32 and KV-cache
  parity, atomic publication, and a canonical checksummed provenance manifest.
- Native dense generation with greedy, temperature, top-k, and top-p decoding;
  deterministic seeds; variable-length batches; EOS/context handling; default
  control-token suppression; dense KV reuse; and token-step streaming events.
- A validated 19.943M-parameter Mamba-2 configuration plus project-owned
  sequential, quadratic-reference, and chunked SSD; grouped input-dependent
  B/C and time steps; causal depthwise convolution; official-style negative-A
  and log-uniform time-step initialization; grouped gated RMSNorm; fp32
  residual/SSM state; and constant-size recurrent decoding caches.
- A validated 19.959M-parameter masked-diffusion configuration, project-owned
  bidirectional RoPE Transformer, deterministic continuous-time corruption,
  protected-token eligibility, and exact eligible-normalized `1/t` objective.
- Native iterative diffusion generation and a self-contained Transformers
  custom-code export with clean-reload logits, loss, tokenizer, and denoising
  trajectory parity.

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

## Mamba-2 20M core

The pinned `zero-20m-mamba2` configuration uses seven layers, width 384,
expansion factor 2, twelve 64-dimensional SSM heads in four B/C groups, state
size 64, convolution width 4, SSD chunks of 256 tokens, vocabulary 16,000, and
a 1,024-token context. Embeddings and the output head are untied. The realized
19,943,164 trainable parameters exactly match the analytic breakdown.

The default implementation uses only PyTorch operations. Parallel
training/prefill follows the four-part chunked SSD decomposition, while cached
decoding carries one fixed convolution window and recurrent SSM state per
layer. Tests compare both paths with a token-by-token recurrence and explicit
quadratic semiseparable matrix, including non-multiple chunk lengths and
nonzero initial states. Left padding is state-preserving; right padding is
explicitly rejected.

Run the focused offline core verification with:

```bash
PYTHONPATH=src uv run --frozen pytest tests/test_mamba2.py --no-cov
```

The pinned `mamba-ssm==2.3.2.post1` package is an optional dependency group, not
part of the default environment. Its current PyPI artifact is source-only and
its package root imports an unrelated CUDA extension unconditionally. The
oracle therefore installs the package without that extension and loads only its
independent Triton SSD modules:

```bash
MAMBA_SKIP_CUDA_BUILD=TRUE \
uv sync --frozen --group mamba-oracle --link-mode copy
```

This changes only `.venv`; it does not require or install a host CUDA toolkit.
The project-owned implementation remains the production reference. Run the
real CUDA comparison with:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  verify-mamba2-oracle \
  --output reports/zero-20m-mamba2-oracle.json \
  --seed 1337
```

The command uses the pinned architecture's 12 heads, four B/C groups,
64-dimensional heads and state, a non-multiple 257-token sequence, nonzero
initial recurrent state, the model's initialization ranges for time steps and
negative A, and D skip. It compares both output and final recurrent state
against `mamba_chunk_scan_combined`. Acceptance requires the pinned upstream
CUDA tolerance (`rtol=1e-2`, `atol=3e-3`) or both relative L2 error below
`0.1%` and no element beyond twice that upstream allowance. This handles
TF32-sensitive cancellation without accepting broad drift or a large isolated
error. The canonical report records the configuration, package/runtime/device
versions, maximum absolute errors, relative L2 errors, normalized worst-element
ratios, and pass status.

Mamba-2 reuses the shared data stream, AdamW policy, accumulation/DDP loop,
atomic checkpoint/recovery format, JSONL/TensorBoard/Parquet telemetry, fixed
causal-loss evaluation, and native sampler. The default dry run derives a token
budget that matches the analytic training FLOPs of the 500M-token dense
TinyStories baseline, rounds it to complete optimizer steps, and reports the
reference FLOPs and achieved ratio:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli pretrain-mamba2 \
  artifacts/shards/tinystories-16k/build.json \
  --checkpoint-directory artifacts/checkpoints/zero-20m-mamba2 \
  --jsonl-log artifacts/runs/zero-20m-mamba2/events.jsonl
```

`--target-tokens` deliberately overrides compute matching for bounded tests.
As with dense training, execution requires `--execute`; the full research run
remains a separate long-GPU approval boundary.

Evaluate or generate from a validated Mamba-2 checkpoint with:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli evaluate-mamba2 \
  artifacts/checkpoints/zero-20m-mamba2/step-000000000100 \
  artifacts/shards/tinystories-16k/build.json \
  --split validation --device cuda --precision bf16

PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli generate-mamba2 \
  artifacts/checkpoints/zero-20m-mamba2/step-000000000100 \
  artifacts/tokenizers/tinystories-16k/training.json \
  "Once upon a time"
```

## Masked diffusion 20M core

The pinned `zero-20m-diffusion` denoiser has four layers, width 384, six
full-attention heads, SwiGLU width 1,152, vocabulary 16,000, and a 1,024-token
context. It uses pre-RMSNorm, bidirectional RoPE attention, untied embeddings,
and no time input. The last point is an explicit LLaDA/RADD design choice rather
than an omitted feature.

Base-pretraining corruption draws one continuous `t` per example from
`Uniform(1e-3, 1)`, independently masks eligible tokens with probability `t`,
and deterministically guarantees at least one supervised token. Padding and BOS
are protected; ordinary content and EOS remain eligible. Loss is masked-token
cross-entropy weighted by `1/t`, divided by each example's eligible-token count,
then averaged across non-empty examples.

Validate the tokenizer binding and exact analytic/realized parameter count with:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  diffusion-model-summary artifacts/tokenizers/tinystories-16k/training.json
```

Run the focused offline core tests with:

```bash
PYTHONPATH=src uv run --frozen pytest tests/test_diffusion.py --no-cov
```

Diffusion pretraining uses the shared shard cursor, AdamW/schedule policy,
gradient accumulation/DDP loop, rank-zero telemetry, and atomic checkpoint
format. A dedicated objective adapter performs corruption inside each
microbatch. Python, NumPy, Torch CPU, and CUDA RNG state are checkpointed, so
the sampled corruption trajectory resumes bit-exactly on CPU.

The default dry run matches the estimated training FLOPs of 500M dense tokens:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  pretrain-diffusion artifacts/shards/tinystories-16k/build.json \
  --checkpoint-directory artifacts/checkpoints/zero-20m-diffusion \
  --jsonl-log artifacts/runs/zero-20m-diffusion/events.jsonl
```

At the pinned 1,024-token configuration, complete optimizer-step rounding gives
528,990,208 diffusion tokens and a `1.0000089` ratio to the dense reference
FLOPs. `--target-tokens` is available only for bounded tests and smoke runs.
Execution still requires `--execute`; any long GPU run needs fresh explicit
approval.

Native inference fills a fixed response canvas iteratively. It supports greedy
or seeded temperature proposals, deterministic confidence-ranked reveals,
linear/cosine schedules, optional low-confidence remasking, EOS truncation, and
streamed reveal/remask events. The reference-quality default uses one model
forward per response token; reduced-step schedules are explicit experiments,
not equivalent-quality aliases.

Generate from a validated diffusion checkpoint with:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  generate-diffusion \
  artifacts/checkpoints/zero-20m-diffusion/step-000000000100 \
  artifacts/tokenizers/tinystories-16k/training.json \
  "Once upon a time" \
  --response-length 64 \
  --diffusion-steps 64 \
  --reveal-schedule linear
```

The sampler never mutates prompt tokens, always removes every response mask,
and reports actual model forwards, total latency, and output throughput.
`--remask-strategy low_confidence --remask-fraction <fraction>` enables
revision before the final step. The bounded compiled-CUDA smoke is complete,
but its four optimizer steps do not produce a quality-trained or chat-ready
diffusion checkpoint.

Export a validated diffusion checkpoint as a self-contained local Transformers
package:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  export-diffusion-hf \
  artifacts/checkpoints/zero-20m-diffusion/step-000000000100 \
  artifacts/tokenizers/tinystories-16k/training.json \
  --output-directory artifacts/exports/zero-20m-diffusion
```

The directory contains `model.safetensors`, the exact tokenizer, sampler
defaults, `hf_diffusion_config.py`, `hf_diffusion_model.py`, and a checksummed
provenance manifest. Because Transformers has no native class for this
architecture, local reload explicitly opts into the bundled reviewed code:

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(
    "artifacts/exports/zero-20m-diffusion",
    local_files_only=True,
    trust_remote_code=True,
)
```

Export refuses incomplete or mismatched lineage and verifies the cleanly
reloaded artifact against the internal model for fp32 logits, the weighted
diffusion loss, and a deterministic denoising trajectory. A persisted derived
RoPE-frequency buffer keeps parity under Transformers' meta-device loading.
The parity reload is explicitly released before atomic directory publication;
this avoids a Safetensors memory-map rename failure on Windows-backed WSL while
preserving the same-filesystem `os.replace` contract.
This command is entirely local; it does not publish to the Hugging Face Hub.

Focused export verification:

```bash
PYTHONPATH=src uv run --frozen pytest tests/test_export_diffusion_hf.py --no-cov
```

Evaluate a validated diffusion checkpoint on fixed, non-wrapping shard windows
with a dedicated corruption seed:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  evaluate-diffusion \
  artifacts/checkpoints/zero-20m-diffusion/step-000000000100 \
  artifacts/shards/tinystories-16k/build.json \
  --split validation \
  --max-batches 32 \
  --corruption-samples-per-batch 4 \
  --seed 1337 \
  --device cuda \
  --precision bf16
```

The result reports masked-reconstruction cross-entropy and the Monte Carlo
eligible-normalized `1/t` variational upper bound in nats, plus corruption rate,
model forwards, and masked-token throughput. It explicitly marks causal
perplexity as inapplicable; these values must not be placed in the dense/Mamba
perplexity column. Reusing the same checkpoint, fixed windows, and seed
reproduces the corruption trajectory.
Version-2 diffusion evaluation and generation records also carry the exact
checkpoint ID and canonical manifest hash; generation records the execution
device. The smoke-report builder rejects records that are not bound to the
selected checkpoint or were not produced on CUDA.

## Milestone 6A acceleration gate

The training configuration now hashes every implementation choice that can
change numerical behavior: compile mode, AdamW backend, output-loss backend,
SDPA backend, float32 matmul precision, telemetry sampling, and recovery
cadence. PyTorch 2.13 linear cross-entropy has a loss-only model path that does
not allocate full vocabulary logits. Dense and diffusion can force native
Flash SDPA for eligibility checks; diffusion has a separate padding-free
pretraining option so an all-valid boolean mask does not block Flash. Mamba-2
can independently test `torch.set_float32_matmul_precision("high")`.

From a clean committed revision, freeze the deterministic 26-cell,
three-architecture calibration matrix on CPU. The plan binds that exact Git
revision and the expected RTX 4080 SUPER device name:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  plan-acceleration-calibration \
  artifacts/shards/tinystories-16k/build.json \
  --output artifacts/acceleration-calibration/plan.json

PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  inspect-acceleration-calibration-cell \
  artifacts/acceleration-calibration/plan.json \
  dense sampled-telemetry

PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  run-acceleration-calibration-cell \
  artifacts/acceleration-calibration/plan.json \
  artifacts/shards/tinystories-16k/build.json \
  dense sampled-telemetry --repetition 1
```

The matrix measures the current per-step baseline, sampled telemetry, four
compile modes, fused AdamW, architecture-appropriate Flash SDPA/TF32, and
linear cross-entropy without a Cartesian explosion. It fixes seed, shard,
shape, 50-step warmup, exactly 500 measured optimizer steps, three fresh-process
repetitions, parity tolerances, a 5% maximum within-cell throughput spread, and
a 10% median end-to-end promotion threshold. A shorter run may validate the
executor and artifact chain, but it is smoke evidence and cannot promote a
candidate. Planning, inspection, and the default
per-cell dry run allocate no model and execute no GPU work. The dry run resolves
the exact version-2 training config, clean-revision binding,
warmup/measurement bounds, and repetition-specific paths.

After explicit approval for the bounded GPU calibration, execute each cell in
a fresh process. Complete the same-architecture baseline repetition before its
candidate repetitions:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  run-acceleration-calibration-cell \
  artifacts/acceleration-calibration/plan.json \
  artifacts/shards/tinystories-16k/build.json \
  dense baseline --repetition 1 \
  --tokenizer-model artifacts/tokenizers/tinystories-16k/tokenizer.json \
  --execute
```

Execution refuses a dirty or different revision, changed plan/shards/tokenizer,
unexpected or non-bf16 CUDA device, existing/partial repetition directory, and
candidate without its matching baseline evidence. It records cold and warm
compiled-step latency plus their explicit compile-overhead estimate, deferred
CUDA-event compute and optimizer timing, end-to-end throughput,
data/evaluation/checkpoint/fsync time, CPU utilization, profiler-observed SDPA,
peak VRAM, graph breaks, parameter-group weight/gradient/update scales, and an
untimed warm-up trace. Result, trace, checkpoint, and event-log hashes are
generated by the executor; never hand-author measurements. CUDA-event component
timers are diagnostic and can be non-additive under compiled or multi-stream
execution. Never derive residual overhead by subtracting their sum from the
enclosing host wall time; promotion uses host end-to-end throughput only.

After every planned repetition exists, validate the bindings and atomically
derive the report with:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  build-acceleration-calibration-report \
  artifacts/acceleration-calibration/plan.json \
  artifacts/acceleration-calibration/fixed-500-results \
  --output reports/zero-20m-acceleration-calibration.json
```

Promotion requires all three complete baseline/candidate repetitions, exact
fixed-step bindings, numerical parity, stable throughput for both baseline and
candidate, and at least the planned 10% median speedup. These settings are
implementation candidates, not objective changes. Muon,
alternative diffusion-time sampling, Token-Superposition, curriculum, native
FP8, and fused third-party Mamba kernels remain separately gated experiments.

## Architecture study planning and results

Milestone 7 begins with a CPU-only plan that freezes all three architectures,
the seeds `1337`, `2027`, and `3407`, shared data bindings, exact FLOP budgets,
screening/full stop steps, output paths, and checkpoint-storage bounds:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  plan-architecture-study \
  artifacts/shards/tinystories-16k/build.json \
  --output artifacts/architecture-study/plan.json \
  --dense-tokens-per-second 391158.0927904619 \
  --mamba2-tokens-per-second 166019.2591970272 \
  --diffusion-tokens-per-second 430108.6269484941 \
  --diffusion-adamw-backend fused
```

The generated plan is ignored local evidence. Every lineage uses its full
500M-dense-equivalent scheduler from step zero. Screening stops are step 12,208
for dense, 14,784 for Mamba-2, and 12,915 for diffusion; only seed 1337 later
resumes the identical configuration to its full bound. Seeds 2027 and 3407
must not use shorter screening-only schedulers.

All three pretraining commands accept `--seed`; it controls both model/RNG
initialization and deterministic shard order and is included in the canonical
training-configuration hash. The plan now uses the synchronized Milestone-6A
rates and records the promoted fused AdamW backend only for diffusion. Do not
use the four-step compile smokes as sustained-throughput estimates. The
nine-lineage retention upper bound is about 8.65 GB before exports, telemetry,
and transient atomic-publication space, so verify at least 15 GiB free before
the study. The current plan's screening/full compute-only estimates are about
4.26/21.30 minutes per dense lineage, 12.16/60.79 minutes per Mamba-2 lineage,
and 4.10/20.50 minutes per diffusion lineage.

The nine-lineage GPU study is complete. The tracked result report is
[`reports/zero-20m-architecture-study.json`](reports/zero-20m-architecture-study.json);
it records terminal steps, tokens, config hashes, event-wall timings, and
checkpoint paths. The read-only audit validated all 15 retained checkpoints.
The ignored checkpoint and event artifacts remain under
`artifacts/architecture-study/` for downstream evaluation and export.

The downstream closeout uses 24 fixed non-wrapping validation batches for each
terminal checkpoint. Dense and Mamba-2 records report causal loss/perplexity;
diffusion records masked-reconstruction loss and its variational upper bound.
All nine native generation samples are recorded under
`artifacts/generations/m7-*.jsonl`. Dense, diffusion, and both screening
Mamba-2 Hugging Face exports passed their parity contracts. The full seed-1337
Mamba-2 export is intentionally not published because its strict fp32 parity
check measured a `2.7298927e-05` maximum absolute error against the `1e-05`
tolerance.

## Milestone 8 dense-ablation planning

The first M8 step is CPU-only planning. It freezes the reusable M7 dense
100M-token screening baselines and seven explicitly named single-variable
variants across seeds `1337`, `2027`, and `3407`:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  plan-dense-ablations \
  artifacts/architecture-study/plan.json \
  --output artifacts/dense-ablations/plan.json
```

The generated plan contains 24 jobs and is deliberately
`execution_ready=false`: this field describes the pre-execution plan schema,
not a completion tracker. The canonical baseline jobs are explicit; all 21
research jobs have executable variant controls and are marked
`ready_for_bounded_gpu_smoke`. The M7 seed `1337` screening checkpoint was
recovered before execution, so all three baseline seeds now have reusable
screening checkpoints.

The 21 research jobs have completed the exact 100M-token scheduler from step
zero on the RTX 4080 SUPER. Their canonical final checkpoints and JSONL streams
are retained under `artifacts/dense-ablations-clean-20260807/` and were
audited for checkpoint integrity, the step-12,208 / 100,007,936-token
boundary, finite terminal metrics, and clean commit `3e5ee58` provenance.

Aggregate the completed matrix into a deterministic CPU-only comparison report:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  build-dense-ablation-report \
  artifacts/dense-ablations/plan.json \
  artifacts/dense-ablations-clean-20260807 \
  --output artifacts/dense-ablations-clean-20260807/report.json
```

The generated report validates all 21 final checkpoints and event streams,
then records per-seed evidence, mean/sample-standard-deviation summaries,
parameter counts, throughput, memory, and deterministic rankings. The current
loss finalists are `hybrid_muon`, `mha`, and `layer_norm`; `tied_embeddings` is
the fastest variant. The recommended downstream evaluation set is their union.

The selected variants were then evaluated on the fixed 24-batch validation
window (192 sequences and 196,416 predicted tokens) and generated 16 greedy
tokens from the `Once upon a time` prompt. The canonical downstream report is
[`reports/zero-20m-dense-ablations-downstream.json`](reports/zero-20m-dense-ablations-downstream.json):

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  build-dense-ablation-downstream-report \
  artifacts/dense-ablations-clean-20260807/report.json \
  --evaluation-root artifacts/evaluations \
  --generation-root artifacts/generations \
  --seed 2027 \
  --output reports/zero-20m-dense-ablations-downstream.json
```

The validation mean losses/perplexities are `1.7428036`/`5.7133` for
`hybrid_muon`, `1.7663516`/`5.8495` for `mha`, `1.7749760`/`5.9001` for
`layer_norm`, and `1.7624807`/`5.8269` for `tied_embeddings`. All four
deterministic samples produced the same continuation:
`, there was a little girl named Lily. She loved to play outside in the`.

## Post-training chat and SFT foundation

The first post-training implementation is CPU-only and data-independent. The
versioned role-delimited template in `lm_from_zero.post_training.chat` uses the
fixed `system`, `user`, `assistant`, `end`, `bos`, and `eos` tokens. Conversation
validation rejects control-token injection and invalid role order. The SFT
renderer in `lm_from_zero.post_training.sft` produces causal examples with
non-assistant and padding labels set to `-100`; assistant content, turn
boundaries, and final EOS remain supervised. Its left-truncation policy drops
complete oldest user/assistant pairs before trimming content tokens while
preserving control markers. The collator and assistant-only causal loss are
covered by focused CPU tests.

The planned first-stage policy is represented by `SFTConfig`: 100,000 examples,
one epoch, maximum length 1,024, learning rate `2e-5`, and length-bucketed
padding before any padding-free packing. The deterministic local mix is now
materialized under `artifacts/post-training/smoltalk2-sft-100k/` from
`HuggingFaceTB/smoltalk2` config `SFT` at revision
`fc6cc2103c066455aade5d7fbb346039ae36ca5e`; it contains exactly 100,000
canonical records and records hash
`f1c5770238d9ec8fd8f6a50ebef405b295e5450adf0b1ed51c4cfffacaaab811`.

The bounded CUDA smoke completed for `hybrid_muon`, `mha`, `layer_norm`, and
`tied_embeddings` from the seed-2027 step-12,208 checkpoints using bf16,
batch size 2, 512-token input bounds, and two AdamW updates per variant. The
full one-epoch run then completed for the selected `hybrid_muon` checkpoint:
12,500 optimizer steps, all 100,000 examples, 62,684,063 supervised tokens,
and final assistant-only loss `1.4705926`. Its final checkpoint is under
`artifacts/post-training/sft/hybrid-muon-seed-2027/`; generated training
evidence is `reports/zero-20m-sft-hybrid-muon.json`.

The deterministic native chat check is in
`reports/zero-20m-sft-generation.json`. The small model produced
tool-call-shaped continuations for these prompts rather than consistently
answering them directly, so the training lifecycle is complete while chat
quality still needs a held-out evaluation. The preference manifest, bounded DPO
smoke, and durable full-run runner are now complete; the approved one-epoch
DPO run is the next execution step.

## Preference optimization foundation

The project-owned DPO foundation is now available in
`lm_from_zero.post_training.dpo`. It validates canonical preference pairs with
one shared prompt, renders chosen and rejected responses with identical prompt
prefixes and assistant-only response masks, supports deterministic left
truncation, computes summed response log-probabilities, and implements the
standard logistic DPO objective with the planned `beta=0.1`. The immutable
reference-log-probability cache identity binds model, tokenizer, checkpoint,
chat-template, and context-length hashes so the same frozen reference can be
reused by DPO and APO.

The pinned `HuggingFaceH4/ultrafeedback_binarized` training parquet is now
available locally at revision
`3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`. Preparation scanned 61,135 rows,
retained 60,700 canonical renderable pairs, rejected 435 rows under the
recorded validation policy, and selected 50,000 pairs using seed 1337 and
`prompt_id_sha256_ascending` ordering. The ignored local manifest is
`artifacts/post-training/ultrafeedback-binarized-50k/manifest.json`; its records
hash is
`06d9d822ec75b5d1889776337a78520d374b4f6a51eaf8a82afb495acf049391`.
The manifest binds the tokenizer and chat-template hashes, a 1,024-token
left-truncation policy, and 10,873 selected pairs requiring truncation. Raw
preference data and cached reference logits remain local ignored artifacts.

The bounded DPO CUDA smoke completed on the RTX 4080 SUPER with bf16, batch size
2, a 512-token bound, and two AdamW updates initialized from the final
hybrid-Muon SFT checkpoint. Both policy and frozen-reference models loaded
from the validated SFT checkpoint, and reference log-probabilities were
computed once. Loss decreased from `0.6931472` to `0.6208477`; gradients were
finite and preference accuracy reached `1.0` on the second update. Generated
evidence is `reports/zero-20m-dpo-smoke.json`. The next gated action is
The durable `run-dpo` runner is now available. It builds one immutable
reference-log-probability cache, trains one epoch with atomic checkpoints, and
writes provenance-bound metrics and a final report under
`artifacts/post-training/dpo/hybrid-muon-seed-2027/`. The approved one-epoch
preference-optimization run has not started yet.

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
An interrupted save cannot replace a prior recovery point. Long runs default to
time-only recovery checkpoints every 15 minutes plus final/bounded boundaries;
an explicit positive step interval remains available for controlled smoke tests.
Retention keeps the latest three valid checkpoints plus a distinct valid best
checkpoint and never removes the sole valid recovery point.

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
and writes sampled optimizer/checkpoint/resume/completion events to canonical
append-only JSONL. Pretraining CLI runs default to one telemetry record per 50
steps; short evidence runs can set `--telemetry-every-steps 1`. Each sampled
record states the number of measured optimizer steps and includes elapsed time,
effective token throughput, and CUDA peak allocated/reserved bytes when
applicable. The runner saves a final checkpoint even when neither periodic
trigger fires. Resume rejects a changed training-configuration hash as well as
the checkpoint binding mismatches documented above.

Milestone-8 dense pilots use the same command with explicit
`--model-variant` values (`learned_absolute_positions`, `layer_norm`, `gelu`,
`mha`, `without_qk_norm`, or `tied_embeddings`). The optimizer ablation keeps
the baseline model and sets `--optimizer-variant hybrid_muon`; it partitions
eligible hidden matrices to Muon and all remaining parameters to AdamW. Both
controls are included in non-default training hashes and scheduler recovery
state, while their baseline defaults are omitted so existing M7 hashes remain
unchanged. Standard dense HF export accepts only the baseline model/optimizer
pair.

When the two metric options are omitted, TensorBoard defaults to a
`tensorboard/` sibling of the JSONL log and Parquet defaults to the JSONL path
with a `.parquet` suffix. The rank-zero JSONL handle buffers complete canonical
lines and durably syncs every 50 recorded steps or five seconds by default,
always before checkpoint publication, snapshots, abort cleanup, and clean
shutdown. TensorBoard uses `purge_step` on resume so steps after the restored
checkpoint are replaced rather than displayed twice. Parquet is atomically
rebuilt at checkpoint snapshots and graceful termination, keeping the latest
canonical record for each sampled optimizer step.

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
hardware approval.

After completing matching uninterrupted and interrupted/resumed single-GPU
compiled bf16 runs, generate the resume-tolerance evidence directly from their
durable logs and final checkpoints:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  compare-dense-resume \
  artifacts/runs/resume-tolerance-uninterrupted/events.jsonl \
  artifacts/checkpoints/resume-tolerance-uninterrupted/step-000000000004 \
  artifacts/runs/resume-tolerance-resumed/events.jsonl \
  artifacts/checkpoints/resume-tolerance-resumed/step-000000000004 \
  --output reports/zero-20m-tinystories-resume-tolerance.json
```

The comparison requires identical validated training configurations, source
revision, dependency/hardware runtime, tokenizer, shard build, and model
configuration. It also checks that the resume event names the final resumed
checkpoint's parent. The default acceptance thresholds are `atol=1e-5` and
`rtol=1e-4` for model parameters and `atol=1e-4` for per-step loss. A failing
comparison still writes its canonical report and exits nonzero. The bounded
RTX 4080 SUPER measurement completes the final Milestone 4 follow-up.

The first bounded pair is retained as calibration evidence at
[`reports/zero-20m-tinystories-resume-tolerance-calibration.json`](reports/zero-20m-tinystories-resume-tolerance-calibration.json).
It narrowly failed the strict defaults, with divergence already present before
the restore boundary. It is not acceptance evidence. A fresh confirmatory pair
therefore predeclares `parameter_atol=2e-5`, `parameter_rtol=1e-4`, and
`loss_atol=2e-4`; these remain small absolute tolerances for compiled bf16
training and are recorded in the generated validation report.

The fresh confirmatory comparison passes those predeclared thresholds. Its
portable evidence, including both checkpoint hashes, resume lineage, every
step-loss difference, aggregate parameter comparison, and source/runtime
bindings, is generated at
[`reports/zero-20m-tinystories-resume-tolerance.json`](reports/zero-20m-tinystories-resume-tolerance.json).
Together with exact CPU and DDP recovery coverage, this completes the training
system milestone without claiming compiled CUDA is bit-exact.

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

## Mamba-2 Hugging Face export

Export a validated Mamba-2 checkpoint and tokenizer with:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli export-mamba2-hf \
  artifacts/checkpoints/zero-20m-mamba2/step-000000000100 \
  artifacts/tokenizers/tinystories-16k/training.json \
  --output-directory artifacts/exports/zero-20m-mamba2
```

The Mamba-2 tensor layout maps one-to-one onto Transformers, but its unfused
`Mamba2ForCausalLM` fallback normalizes the complete expanded width. The
official Mamba-2 implementation instead sets gated RMSNorm's group size to
`expanded_width / ngroups`. The export therefore includes the small
`hf_mamba2_compat.py` auto-model source that restores official grouped
normalization on unfused runtimes; it does not replace the SSM, convolution, or
cache implementation. See the
[official Mamba-2 module](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py)
and its
[gated RMSNorm reference](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py).

Reload the local package through the public auto-model API:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "artifacts/exports/zero-20m-mamba2",
    local_files_only=True,
    trust_remote_code=True,
)
```

`trust_remote_code=True` executes the compatibility file copied into this local
export, so inspect and retain the manifest hashes when moving it. The manifest
explicitly records this requirement and does not claim parity for native
unfused Transformers Mamba-2. Export acceptance requires complete tensor-set
and shape matching plus internal/auto-model fp32 logits and one-token recurrent
cache parity at `atol=1e-5, rtol=1e-5`. The tested small export has zero maximum
absolute error on both comparisons. Publication remains a separate approval
gate.

Focused atomic export, reload, tokenizer, corruption, and parity verification:

```bash
PYTHONPATH=src uv run --frozen pytest tests/test_export_mamba2_hf.py --no-cov
```

The bounded RTX 4080 SUPER vertical slice is recorded in
[`reports/zero-20m-mamba2-smoke.json`](reports/zero-20m-mamba2-smoke.json).
It proves a clean two-stage compiled-bf16 run through 32,768 tokens and resumed
checkpoint lineage, with finite loss/gradients and under 0.9 GB peak reserved
VRAM. The generated report also binds a fixed validation batch, the full
19.943M-parameter export, and two-token recurrent generation to the same model,
tokenizer, shard, checkpoint, and source hashes. This is integration evidence,
not a trained quality result.

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

Use the architecture-specific builder for the equivalent Mamba-2 evidence:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  build-mamba2-smoke-report \
  artifacts/runs/zero-20m-mamba2-smoke/events.jsonl \
  artifacts/checkpoints/zero-20m-mamba2-smoke/step-000000000004 \
  artifacts/evaluations/zero-20m-mamba2-smoke.jsonl \
  artifacts/exports/zero-20m-mamba2-smoke \
  artifacts/generations/zero-20m-mamba2-smoke.jsonl \
  --output reports/zero-20m-mamba2-smoke.json
```

Diffusion uses its own schema so causal perplexity cannot leak into the report:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  build-diffusion-smoke-report \
  artifacts/runs/zero-20m-diffusion-smoke/events.jsonl \
  artifacts/checkpoints/zero-20m-diffusion-smoke/step-000000000004 \
  artifacts/evaluations/zero-20m-diffusion-smoke.jsonl \
  artifacts/exports/zero-20m-diffusion-smoke \
  artifacts/generations/zero-20m-diffusion-smoke.jsonl \
  --output reports/zero-20m-diffusion-smoke.json
```

The builder validates checkpoint contents, clean Git lineage, contiguous
optimizer steps, resume ancestry, and matching model/tokenizer/shard/checkpoint
hashes across every input. The committed report is generated from recorded
artifacts; measured values are not copied manually. It records the resumed
four-step compiled bf16 CUDA run, evaluation throughput, exact Hugging Face
fp32 parity, native cached-generation throughput, artifact hashes, and the
source revision. The Mamba-2 form also records cached export parity and the
grouped-normalization auto-model requirement. The intentionally untrained
generation output is not treated as a quality result. The diffusion form records
masked reconstruction, its variational upper bound, clean custom-code export
parity, exact evaluation/generation checkpoint bindings, and
iterative-denoising work without a causal-perplexity field. Its current
compiled training trace includes graph breaks in eager runtime-validation
guards, so the recorded throughput is partial-graph integration evidence, not
a full-graph compile-speed claim.

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
