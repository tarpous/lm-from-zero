# Session handoff: dense vertical-slice implementation

## Current update (August 4, 2026)

Milestone 6's bounded masked-diffusion CUDA smoke is complete locally. The
default dry run preserved the compute-matched 528,990,208-token scheduler, then
the compiled-bf16 integration run stopped at step 2 and resumed through step 4
for 32,768 total tokens. All four losses and gradient norms were finite; peak
reserved CUDA memory was 1,522,532,352 bytes. One fixed seeded validation batch
measured 9.7700793 nats masked-reconstruction loss, a 9.7017822-nat
eligible-normalized variational upper bound, and a 0.56640625 mask rate.

The step-4 self-contained `LLaDAForMaskedDiffusion` export passed exact fp32
logit and loss parity plus deterministic denoising-trajectory parity. Native
CUDA generation completed an eight-token canvas in eight model forwards. The
portable version-2 evidence is generated at
[`reports/zero-20m-diffusion-smoke.json`](reports/zero-20m-diffusion-smoke.json).
It binds evaluation, export, and generation to the exact step-4 checkpoint ID
and canonical manifest hash, and records CUDA for evaluation and generation.

The first real export exposed a Windows-backed WSL publication bug:
Safetensors kept the temporary model file memory-mapped while the custom-code
reload remained alive, so DrvFS rejected the final atomic directory rename.
The exporter now releases and collects that reload before `os.replace`; the
focused regression suite passes, and the real `/mnt/c` export succeeds.

PyTorch also reported scalar graph breaks in diffusion input/loss validation
guards. `torch.compile` was invoked and the bounded run completed with finite
metrics, but the trace is partial-graph integration evidence rather than a
full-graph compile-speed claim. Do not change capture settings retroactively;
move or specialize the eager validation guards in a separately tested phase if
full-graph compilation becomes an explicit goal.

Milestone 7 is next after the complete offline gate and the separate approved
commit/push boundary. Its three-seed screening and full seed-1337
compute-matched architecture study are long GPU work and require fresh explicit
approval. Milestone 6's four-step checkpoint is not quality-trained or
chat-ready.

The complete post-smoke offline gate passes formatting, lint, strict typing,
CLI and lock discovery, and all 177 tests with 85.56% branch coverage.

## Current update (July 27, 2026)

The Safetensors checkpoint objective described below is complete and the older
instructions are retained only as historical context. Work has advanced through:

- atomic checkpoint/recovery with exact CPU resume and retention;
- the dry-run-first single-process and `torchrun` DDP dense trainer with
  reduced global metrics, rank-zero logs/checkpoints, and rank-local
  cursor/RNG recovery;
- canonical durable JSONL metrics, resume-aware rank-zero TensorBoard scalars,
  and atomic typed Parquet snapshots/rebuilding;
- validated compiled-CUDA resume-tolerance report generation with explicit
  parameter/loss thresholds and nonzero CLI status on failure;
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
lint, strict mypy, CLI/torchrun discovery, lock validation, and 140 tests with
85.71% branch coverage.

The measured dense integration slice is complete. A compiled bf16 CUDA run
used the full-run scheduler configuration, stopped at optimizer step 2, resumed
from that checkpoint through step 4, evaluated one fixed validation batch,
exported a standard local OLMo2 artifact with exact fp32 parity, and completed
cached greedy generation. The portable generated evidence, including
throughput, peak CUDA memory, lineage, source and artifact hashes, lives at
[`reports/zero-20m-tinystories-smoke.json`](reports/zero-20m-tinystories-smoke.json).
This is integration evidence only; four optimizer steps do not establish model
quality.

Milestone 4 is complete. The bounded compiled-CUDA interrupted/resumed versus
uninterrupted comparison validates canonical logs, clean matching checkpoints,
source/runtime/config bindings, resume ancestry, complete step metrics, every
final model tensor, and declared tolerances. The first strict-default pair is
retained as a non-passing calibration report. A fresh confirmatory pair from
clean revision `1f67477d6c985d308defa7c35c43432620ebb309` passes the predeclared
calibrated thresholds in
[`reports/zero-20m-tinystories-resume-tolerance.json`](reports/zero-20m-tinystories-resume-tolerance.json).
This supports tolerance-equivalent compiled CUDA recovery, not bit-exactness.

Milestone 5 has started. The first phase implements the validated pinned
19.943M-parameter Mamba-2 configuration, sequential recurrence, explicit
quadratic SSD reference, project-owned chunked SSD with arbitrary final chunk,
causal depthwise convolution, grouped selective parameters, official-style
time-step/negative-A initialization, grouped gated RMSNorm, fp32
residual/sensitive state, left-padding behavior, and constant-size recurrent
cache. The complete offline gate now passes 140 tests with 85.71% branch
coverage.

The shared lifecycle phase is also complete: Mamba-2 uses the audited AdamW
partition, dry-run-first single-process/DDP runner, atomic checkpoint and
bit-exact CPU resume, canonical telemetry, fixed-window causal evaluation, and
native batched recurrent generation. Its default dry run derives a complete
optimizer-step token budget matching the analytic training FLOPs of the
500M-token dense reference and records the achieved ratio. The complete
offline gate passes 144 tests with 85.37% branch coverage.

The Hugging Face phase is complete with an explicit compatibility boundary.
Every Mamba-2 tensor maps one-to-one to Transformers, but its unfused
`Mamba2ForCausalLM` normalizes the full expanded width while the official
Mamba-2 implementation uses `expanded_width / ngroups`. The atomic export
therefore includes a small auto-model source that restores grouped gated
RMSNorm and requires `trust_remote_code=True` on reload. Its manifest explicitly
does not claim native-unfused Transformers parity. Internal/auto-model fp32
logits and one-token recurrent-cache parity both pass with zero maximum absolute
error in the small test model. The complete offline gate passes 147 tests with
85.25% branch coverage.

The bounded compiled-bf16 CUDA vertical slice is complete on the RTX 4080 SUPER.
It trained in two stages through step 4 and 32,768 tokens, restored exact
checkpoint lineage from step 2, kept finite loss/gradients, and stayed below
0.9 GB peak reserved VRAM. A fixed validation batch reached 2,875 predicted
tokens/s. The full 19.943M-parameter export passed fp32 full-logit and cached
parity with maximum absolute errors `1.13e-6` and `8.94e-7`, followed by native
recurrent generation. All hashes and measurements are generated in
`reports/zero-20m-mamba2-smoke.json`.

Milestone 5 is now complete. The optional group pins
`mamba-ssm==2.3.2.post1`; because its current PyPI artifact is source-only, it
is installed with `MAMBA_SKIP_CUDA_BUILD=TRUE` and the harness namespace-loads
only the independent Triton SSD modules instead of the package root's unrelated
`selective_scan_cuda` import. The real 12-head/four-group, 257-token comparison
passed on the RTX 4080 SUPER with nonzero initial state and the model's actual
time-step/A initialization domain. Output and final-state relative L2 errors
were `0.00074075` and `0.00070149`; no element exceeded the bounded acceptance
policy. The complete machine-readable evidence is in
[`reports/zero-20m-mamba2-oracle.json`](reports/zero-20m-mamba2-oracle.json).
The post-oracle complete gate passes formatting, lint, strict typing, CLI and
lock discovery, and all 152 tests with 85.48% branch coverage.

The next implementation phase is Milestone 6, the compute-matched masked
diffusion model. The 500M-token dense and 605.5M-token compute-matched Mamba
runs remain long GPU jobs and require fresh explicit approval. Data downloads,
publication, and other external changes retain their own approval gates.

Milestone 6's core phase is implemented locally: a resolved
19.959M-parameter configuration, project-owned four-layer bidirectional RoPE
Transformer, deterministic continuous-time mask corruption, BOS/padding
protection with EOS eligibility, guaranteed per-example supervision, and the
eligible-normalized `1/t` masked objective. Focused tests cover exact
parameter/FLOP accounting, mask-rate statistics, deterministic corruption,
direct numerical loss, right-context influence, padding, finite gradients, and
invalid contracts. The complete post-core gate passes 161 tests with 85.23%
branch coverage plus formatting, lint, strict typing, CLI discovery, and lock
validation. The next phase is shared training/checkpoint integration; the
diffusion sampler, export, and CUDA smoke follow after that.

The shared training/checkpoint phase is now implemented locally. A diffusion
objective adapter corrupts each microbatch without branching through denoiser
internals, and the existing single-process/DDP loop, optimizer, scheduler,
metrics, cursor, atomic checkpoint, retention, and RNG recovery paths are
reused. A focused interrupted/resumed CPU run exactly matches uninterrupted
parameters and loss, proving the corruption RNG trajectory is recovered. The
real dry run resolves 528,990,208 tokens, 64,574 optimizer steps, and
53,821,918,276,485,120 estimated training FLOPs: a `1.0000089` ratio to the
500M-token dense reference after complete-step rounding. `--execute` remains a
fresh long-GPU approval boundary. The complete post-integration gate passes
162 tests with 85.15% branch coverage plus formatting, lint, strict typing, CLI
discovery, and lock validation.

The native sampler phase is implemented locally. It supports batched
variable-length prompts with fixed response canvases, linear/cosine reveal
schedules, reference full-step and explicit reduced-step denoising, greedy or
seeded temperature proposals, deterministic confidence tie-breaking, optional
low-confidence remasking, immutable prompts, EOS truncation, streamed
reveal/remask events, canonical JSONL evidence, and measured forward/latency
counts. Focused tests cover every schedule/strategy, remasking, prompt
immutability, mask-free termination, seed behavior, EOS, model-mode restoration,
invalid contracts, evidence append, and CLI checkpoint/tokenizer dispatch.
The complete post-sampler gate passes 169 tests and 12 subtests with 85.36%
branch coverage, plus formatting, lint, strict typing, CLI discovery, and lock
validation.

The diffusion Hugging Face phase is implemented locally. It exports a
self-contained `LLaDAForMaskedDiffusion` package with explicit tensor mapping,
Safetensors, exact tokenizer metadata, mask/sampler defaults, bundled
configuration/model source, checksums, and atomic publication. Reload requires
the documented `trust_remote_code=True` boundary because Transformers has no
native diffusion class. The exporter reloads the completed temporary artifact
before publication and requires fp32-logit, eligible-normalized diffusion-loss,
and deterministic denoising-trajectory parity. A regression test persists the
derived RoPE frequencies in the export so Transformers meta-device loading
cannot silently replace the non-checkpoint buffer.
The complete post-export gate passes 172 tests and 12 subtests with 85.48%
branch coverage, plus formatting, lint, strict typing, CLI discovery, and lock
validation.

The architecture-specific diffusion evaluator is implemented locally. It uses
fixed non-wrapping shard windows and a dedicated seeded corruption generator,
then records masked-reconstruction cross-entropy, the eligible-normalized
`1/t` variational upper bound, mask rate, model forwards, throughput, hashes,
and exact cursor movement. Its schema explicitly marks causal perplexity as
inapplicable. Tests cover deterministic repetition, checkpoint/CLI/JSONL
integration, mismatched lineage, epoch wrapping, unavailable CUDA, and invalid
timing.
The complete post-evaluation gate passes 174 tests and 12 subtests with 85.62%
branch coverage, plus formatting, lint, strict typing, CLI discovery, and lock
validation.

The diffusion smoke-report builder is implemented locally. It requires one
clean compiled-bf16 CUDA start/resume lineage and cross-validates the final
checkpoint against the diffusion evaluation, self-contained export, and native
generation records. Its portable schema contains diffusion loss/bound,
corruption, custom-code parity, denoising-forward, throughput, runtime, and
artifact-hash evidence and intentionally has no validation-perplexity field.
The complete post-builder gate passes 175 tests and 12 subtests with 85.49%
branch coverage, plus formatting, lint, strict typing, CLI discovery, and lock
validation.

### Pause point (July 28, 2026)

The user paused immediately before the bounded diffusion CUDA smoke. No
`pretrain-diffusion` process is running, and there are no local diffusion
checkpoint, run, evaluation, export, or generation artifacts yet. The clean
source revision at the pause is
`ca8de45bd24e7cec94a52de2cfae5805ed5ea6b5`, synchronized with
`origin/main`. The live preflight saw `Ubuntu-24.04` under WSL2 and an NVIDIA
GeForce RTX 4080 SUPER with 16,376 MiB, driver 595.79.

Resume with a short two-stage compiled-bf16 CUDA integration run. Keep the
default compute-matched full-run token target and scheduler; do **not** pass
`--target-tokens`. The stop bounds exercise only two and then four optimizer
steps without pretending that the smoke is quality training.

First run the command without `--execute` and inspect its plan, then add
`--execute`:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  pretrain-diffusion artifacts/shards/tinystories-16k/build.json \
  --checkpoint-directory artifacts/checkpoints/zero-20m-diffusion-smoke \
  --jsonl-log artifacts/runs/zero-20m-diffusion-smoke/events.jsonl \
  --tensorboard-directory artifacts/runs/zero-20m-diffusion-smoke/tensorboard \
  --parquet-log artifacts/runs/zero-20m-diffusion-smoke/metrics.parquet \
  --stop-after-optimizer-step 2
```

After the bounded first stage succeeds, resume the identical configuration:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  pretrain-diffusion artifacts/shards/tinystories-16k/build.json \
  --checkpoint-directory artifacts/checkpoints/zero-20m-diffusion-smoke \
  --jsonl-log artifacts/runs/zero-20m-diffusion-smoke/events.jsonl \
  --tensorboard-directory artifacts/runs/zero-20m-diffusion-smoke/tensorboard \
  --parquet-log artifacts/runs/zero-20m-diffusion-smoke/metrics.parquet \
  --resume-from \
    artifacts/checkpoints/zero-20m-diffusion-smoke/step-000000000002 \
  --stop-after-optimizer-step 4 \
  --execute
```

The first-stage command also needs `--execute` only after its dry-run output has
been reviewed. On success, evaluate step 4 with `evaluate-diffusion` on one
fixed validation batch and seeded corruption, export it with
`export-diffusion-hf`, generate a short CUDA response with
`generate-diffusion --response-length 8 --diffusion-steps 8`, and write each
JSONL record under the matching ignored `artifacts/evaluations`,
`artifacts/generations`, and export paths. Finally run
`build-diffusion-smoke-report` to generate
`reports/zero-20m-diffusion-smoke.json`, rerun the full offline gate, update the
README/handoff from generated measurements, and commit/push the completed
Milestone 6 smoke phase.

This smoke is the next action. It is not the long compute-matched architecture
run: the three-seed screening and full seed-1337 budget remain a fresh explicit
long-GPU approval boundary.

The master plan now ends with a separate final-readiness stage. After all
architecture, training, post-training, export, evaluation, and serving
milestones, every selected model must have a real approved training lineage,
architecture-appropriate instruction/post-training, held-out chat-quality
evidence, a cleanly reloadable artifact, and a documented local RTX 4080 SUPER
chat launch path. Smoke checkpoints and base next-token models must never be
presented as genuinely ready. The final handoff must list each ready model,
where it lives, how to launch it, and the evidence and limitations attached to
it.

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
