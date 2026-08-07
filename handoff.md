# Session handoff: dense vertical-slice implementation

## Session closeout (August 7, 2026, M8 execution closeout)

The pushed M8 downstream revision `64a4631` includes pinned `tqdm` live progress
reporting, variant-aware dense evaluation/generation, canonical M8 aggregation,
and downstream evidence reporting. Its complete offline gate passed Ruff,
strict mypy, 249 tests, and 85.01% total branch coverage. The source revision
is on `origin/main`; ignored run artifacts remain local.

This closeout now also includes the selected-variant downstream evidence. The
canonical report is
`reports/zero-20m-dense-ablations-downstream.json`, built from seed `2027` with
24 fixed validation batches (192 sequences, 196,416 predicted tokens) and 16
greedy generation tokens for each selected variant. Mean validation loss and
perplexity were `1.7428036`/`5.7133` for `hybrid_muon`, `1.7663516`/`5.8495`
for `mha`, `1.7749760`/`5.9001` for `layer_norm`, and `1.7624807`/`5.8269`
for `tied_embeddings`. Every sample produced the same continuation:
`, there was a little girl named Lily. She loved to play outside in the`.

The post-training implementation under `src/lm_from_zero/post_training/` now
includes the versioned role-delimited chat template, conversation validation,
assistant-only SFT labels, left truncation, right padding, deterministic
SmolTalk2 mix preparation, and a generated CUDA smoke runner. The local mix
contains exactly 100,000 canonical records at pinned revision
`fc6cc2103c066455aade5d7fbb346039ae36ca5e`; its records hash is
`f1c5770238d9ec8fd8f6a50ebef405b295e5450adf0b1ed51c4cfffacaaab811`.

The bounded CUDA smoke completed for the four selected seed-2027 M8
checkpoints (`hybrid_muon`, `mha`, `layer_norm`, and `tied_embeddings`) with
bf16, batch size 2, a 512-token bound, and two AdamW updates per variant. All
losses and gradient norms were finite, and each variant improved its measured
loss on the second update. Generated evidence is
`reports/zero-20m-sft-smoke.json`.

The approved full SFT run completed for the selected `hybrid_muon` checkpoint
using the baseline model variant and AdamW: one epoch, 12,500 optimizer steps,
100,000 examples, 62,684,063 supervised tokens, and final assistant-only loss
`1.4705926` (mean of the last 100 steps `1.4433360`). Its durable run manifest,
metrics, recovery checkpoints, and final step-12,500 model are under
`artifacts/post-training/sft/hybrid-muon-seed-2027/`; the generated summary is
`reports/zero-20m-sft-hybrid-muon.json`. The final checkpoint and all 12,500
optimizer metrics passed structural, hash, contiguity, and finite-value
validation.

The deterministic native chat check is
`reports/zero-20m-sft-generation.json`. On three fixed prompts, the 20M model
mostly emitted tool-call-shaped continuations rather than direct answers. This
is the first behavior result for the SFT lifecycle; held-out chat evaluation
is still required before making a quality claim. The next gate is approval to
publish this milestone and/or download the preference data for DPO.

The M8 bounded GPU smoke completed for all 21 variant jobs. Every step-4
checkpoint passed structural and cryptographic validation, and every JSONL
stream ended at step 4 with finite loss, gradient, throughput, and memory
metrics.

The canonical strict replacement then ran all 21 jobs from step zero under
`artifacts/dense-ablations-clean-20260807/` and reached the exact
12,208-optimizer-step / 100,007,936-token boundary. Every final checkpoint and
event stream passed the read-only audit: all 21 final manifests bind clean
revision `3e5ee58`, all include a step-zero `run_start` and a `run_complete`
event at the exact boundary, and every terminal loss, gradient norm, and
throughput value is finite.

The missing M7 dense seed-1337 screening checkpoint was recovered by replaying
the exact 500M-token scheduler to step 12,208, then copied into the canonical
`artifacts/architecture-study/dense/seed-1337/checkpoints/step-000000012208`
directory and validated. The regenerated M8 plan marks all three dense
baseline seeds as `reuse_m7_screening`; its `execution_ready` flag remains
false by plan-schema contract even though the 21 full runs are complete. The
CPU-only aggregation now validates all 21 jobs and writes
`artifacts/dense-ablations-clean-20260807/report.json`. Its generated report
selects `hybrid_muon`, `mha`, and `layer_norm` by mean terminal loss, identifies
`tied_embeddings` as fastest, and recommends the four-variant union for
downstream evaluation. Fixed-window validation and native generation for that
union are complete. The full SFT lifecycle now has a completed training run,
checkpoint, and native generation evidence. The next external gate is
approval to publish this milestone and/or download the preference data for
DPO.

## Current update (August 5, 2026, Milestone 7 downstream closeout)

The nine canonical Milestone-7 terminal checkpoints now have fixed-window
validation records under `artifacts/evaluations/m7-*-v24.jsonl`. Each record
uses 24 validation batches (192 sequences, 196,608 source tokens) so the
validation shard is never wrapped or repeated. Dense and Mamba-2 use causal
loss/perplexity; diffusion records masked-reconstruction loss and its
variational upper bound and correctly has no causal perplexity field.

All nine checkpoints also have deterministic short native-generation records
under `artifacts/generations/m7-*.jsonl` for the fixed `Once upon a time`
prompt. Dense and diffusion exports, plus both screening Mamba-2 exports,
passed their Hugging Face export contracts under `artifacts/exports/m7-*`.
The full seed-1337 Mamba-2 export remains intentionally unpublished because
its existing fp32 parity gate rejected a maximum absolute logit error of
`2.7298927e-05` against the `1e-05` tolerance; no tolerance was weakened.

The first 32-batch validation attempt was rejected before producing a record
because it would wrap the finite validation shard. The corrected 24-batch
window is the canonical downstream evidence. The CPU-only M8 dense-variant
contract remains generated at `artifacts/dense-ablations/plan.json`. It
contains 24 jobs: the three baseline jobs are represented explicitly and all
21 research variants have executable controls. The model variants, hybrid
Muon/AdamW partition, runner controls, checkpoint layout, and standard export
rejection are implemented and CPU-tested. The clean 21-job execution and
validation are recorded in the August 7 closeout above; `execution_ready`
remains false as a pre-execution plan-schema value.

All three M7 dense baseline seeds are now reusable, including the restored
seed-1337 screening checkpoint at step 12,208 under the identical M7 schedule.

The default dense training configuration was revalidated against the recorded
M7 seed-1337 event stream: its current canonical hash remains
`0a8bda0ec01938c571f1aa3b78ccd06ee1193153307da806b5393bd75723cde1`.
The variant-aware dry-run path was exercised for GELU and hybrid Muon. A
two-minute CPU execution attempt on the full 20M GELU model did not complete a
step and produced no checkpoint; use the RTX 4080 SUPER for the required
bounded smoke rather than treating CPU throughput as evidence.

## Current update (August 5, 2026, local judge selection)

The local judge decision is now fixed to two models and excludes gpt-oss-20b:

- Primary candidate: Qwen3.5-9B through llama.cpp, preferably the pinned
  Qwen-calibrated imatrix `Q6_K_L` or `Q8_0` GGUF.
- Independent calibration candidate: official Gemma 4 12B instruction-tuned
  QAT Q4 GGUF.

The 27B Qwen3.5 variants were rejected as the normal primary judge for the
RTX 4080 SUPER 16GB target. Standard Q4 files leave insufficient runtime and
KV-cache headroom; specialized IQ4/KV-cache variants are constrained and are
not the reproducible default. M12 should compare the two selected models on
the fixed 100-prompt calibration subset before scoring all 500 prompts. No
judge weights or new dependencies have been downloaded.

## Current update (August 5, 2026, Milestone 7 GPU study complete)

The complete nine-lineage architecture study ran at clean revision
`53d071b46d600cfc50b53ab09f0a5611e9f28f8c` using the frozen plan at
`artifacts/architecture-study/plan.json`. Dense, Mamba-2, and diffusion each
ran seeds 1337/2027/3407 with the full scheduler preserved from step zero;
only seed 1337 continued from screening to the full matched budget. Diffusion
used the promoted fused AdamW backend for every seed.

All nine terminal runs reached their planned boundaries with matching model,
training-config, tokenizer, shard, seed, and token bindings:

- Dense: screening steps 12,208 for seeds 2027/3407; seed 1337 full step
  61,036 and 500,006,912 tokens.
- Mamba-2: screening steps 14,784 for seeds 2027/3407; seed 1337 full step
  73,919 and 605,544,448 tokens.
- Diffusion: screening steps 12,915 for seeds 2027/3407; seed 1337 full step
  64,574 and 528,990,208 tokens.

The tracked result is [`reports/zero-20m-architecture-study.json`](reports/zero-20m-architecture-study.json).
Its read-only audit validated all 15 retained checkpoints, model/recovery
hashes, and terminal manifests. Canonical artifacts occupy about 3.88 GB;
the `/mnt/c` filesystem had about 140 GB free at audit. One initial 60-second
wrapper timeout was moved to the explicitly excluded
`dense/seed-1337-timeout-20260805` evidence directory; the canonical rerun
completed normally.

The next phase is result analysis and downstream evaluation, not more
architecture-study training. Do not delete the ignored checkpoints before
export/evaluation and publication decisions are recorded.

## Current update (August 5, 2026, Milestone 6A calibration complete)

The fixed-step acceleration calibration is complete on the RTX 4080 SUPER at
clean revision `d69351425a8f8b6e3fe3e2e07836c8d007161f4e`. The canonical report
is [`reports/zero-20m-acceleration-calibration.json`](reports/zero-20m-acceleration-calibration.json)
and contains 78 validated results: 26 cells across dense, Mamba-2, and
diffusion, with three fresh-process repetitions per cell. Every result has the
exact 50 warm-up plus 500 measured-step contract, complete checkpoint/trace/
event/manifest hashes, and matching plan, model, tokenizer, shard, seed, and
revision bindings. The 5% stability gate passed for every reported cell.

Frozen outcomes under the predeclared 10% end-to-end speedup and numerical
parity gates:

- Dense: no promotion. The fastest candidates were max-autotune-no-cudagraphs
  (+18.91%) and max-autotune (+16.98%), but both failed loss/gradient parity.
- Mamba-2: no promotion. Max-autotune was fastest (+6.76%) but below the speed
  gate; fused AdamW was +4.56% and narrowly failed loss parity.
- Diffusion: `fused-adamw` is promoted at +13.61% with parity passing. Its
  median optimizer time fell about 58.6% with effectively unchanged memory.

Diffusion Flash-SDPA was proven eligible and faster, but failed parity; linear
cross-entropy was fastest and reduced memory, but failed gradient parity. Keep
the dense and Mamba compiled-default baselines, and use fused AdamW only for
the diffusion track unless a separately approved parity study changes these
decisions. No Milestone-7 long training run was started by this calibration.

## Current update (August 5, 2026, Milestone 6A dense executor smoke)

Milestone 6A's implementation was committed and pushed as
`d1a8c17f5fa44184036c06b5862338fb2a9ef721`. The clean-revision calibration
plan bound that commit and the RTX 4080 SUPER, and one explicitly approved
`dense/baseline/repetition-01` execution completed its 50 warm-up plus 100
measured optimizer steps. Its artifacts, hashes, 102-event log, numerical
trace, and step-150 recovery checkpoint all validate. The measured interval was
1.87462058 seconds for 819,200 tokens (436,995 tokens/second), with Flash SDPA,
zero graph breaks, and finite loss/gradient evidence.

This result is retained as instrumentation smoke evidence only and must not be
mixed into promotion reporting. Its window was too short for the 10% gate.
CUDA-event compute was 1.75679 seconds and optimizer timing was 0.23408 seconds;
their 1.99087-second sum exceeds the enclosing host interval because compiled
or multi-stream component timers are non-additive. Do not calculate residual
overhead from those components.

The corrective offline phase now fixes the plan at exactly 500 measured steps
after 50 warm-up steps, requires exact step binding, and adds a 5% maximum
relative throughput-spread gate across all three fresh-process repetitions.
Its full offline gate passes Ruff formatting/lint, strict mypy across 74 files,
all 224 tests, and 85.05% total branch coverage. Publish this correction as the
next revision after `d1a8c17`, regenerate a clean-revision plan under the fresh
`fixed-500-results` root, and request approval before the replacement dense
baseline GPU run. Do not start candidates or Milestone 7.

## Current update (August 5, 2026, Milestone 6A implementation gate)

The previous architecture-study planner revision is pushed: local `HEAD` and
`origin/main` both began this phase at `976e80479347f5c16744caf98fb5773b4068a97b`.
Milestone 6A's executor is now implemented locally through the CPU/offline
gate, but the milestone is not yet complete because no synchronized RTX 4080
SUPER calibration results have been generated.

The training configuration is version 2 and hashes explicit compile mode,
AdamW backend, full-logit versus PyTorch linear cross-entropy loss, SDPA
backend, float32 matmul precision, telemetry interval, metric durability, and
checkpoint cadence. The runner removes unconditional per-step CUDA
synchronization, samples scalar/peak-memory telemetry by window, uses time-only
15-minute recovery checkpoints by default, durably flushes buffered canonical
JSONL before every checkpoint and shutdown, and preserves step cadence for
bounded smoke tests. Dense and diffusion can force native Flash SDPA;
diffusion has an explicit padding-free attention path. Mamba-2 can test TF32.
Linear loss-only forwards preserve the checkpoint/export parameter layout and
have CPU loss/gradient parity tests for all three implemented architectures.

`src/lm_from_zero/acceleration_calibration.py` defines a deterministic 26-cell
matrix for dense, Mamba-2, and diffusion: compiled-default current baseline,
sampled telemetry, compile-disabled plus three alternative compile modes,
fused AdamW, and architecture-appropriate Flash SDPA, TF32, and linear
cross-entropy candidates. The version-2 plan fixes the clean Git revision,
expected GPU name, seed, shard/model hashes, shape, warmup, measurement window,
three repetitions, loss/gradient/update parity tolerances, and a 10% median
end-to-end promotion threshold. Plan generation now refuses a dirty worktree.
Use after committing this implementation:

```bash
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli \
  plan-acceleration-calibration \
  artifacts/shards/tinystories-16k/build.json \
  --output artifacts/acceleration-calibration/plan.json
```

Planning, inspection, and the default `run-acceleration-calibration-cell` dry
run are CPU-only and do not allocate a model. The `--execute` path is now real
and guarded: it verifies the clean planned revision, exact plan/shard/tokenizer,
unused repetition directory, expected bf16 CUDA device, and same-repetition
baseline evidence before candidate allocation. It writes generated numerical
traces and result hashes, uses CUDA events with one resolution boundary,
profiles SDPA outside the measured window, measures graph breaks, CPU/data,
evaluation, checkpoint, JSONL fsync and peak VRAM, and records decay/no-decay
weight, gradient, update, angular-LR and effective-LR statistics. Baseline
parity uses the first three synchronized warm-up losses/gradient norms plus
warm-up update RMS; candidates bind the matching baseline result hash.

The next action is to commit/push this implementation, regenerate the
clean-revision plan, inspect an exact cell dry run, and request explicit
approval before any `--execute` call. Run baseline repetitions before their
candidates. Do not hand-author result/report values, do not promote a candidate
without complete parity evidence and at least 10% median end-to-end gain, and
do not start the nine Milestone-7 screening lineages.

The complete offline gate passes all 223 tests with 85.05% branch coverage;
the approval-gated CUDA execution body is explicitly outside the CPU-only
coverage denominator, while its guards, dispatch, schemas, event timer,
profiler classifier and statistic reducers are covered. Ruff formatting/lint
and strict mypy pass across all 74 source/test files. No dependency was
installed, no dataset/model was downloaded, and no GPU training was executed
in this implementation phase.

## Current update (August 4, 2026, acceleration-plan refresh)

The external implementation plan at
`../plans/01-lm-from-zero.md` now contains a
primary-source-verified training-acceleration program refreshed through August
4, including Qwen3.5/3.6/3.7, MiniMax M3, GLM-5.2, Nemotron 3 Ultra and
Nemotron-Labs-Diffusion, Thinking Machines Inkling, PrismML Bonsai, and Kimi
K3. It distinguishes real training gains from inference-only, long-context,
cluster-only, and non-reproducible claims.

Before the long Milestone-7 architecture runs, implement and verify Milestone
6A from that plan: remove hot-path host/CUDA synchronization, rationalize
checkpoint/evaluation durability, benchmark compile modes/fused AdamW/SDPA/
linear cross-entropy/microbatch choices, preserve compiler caches, and produce
synchronized end-to-end calibration records for dense, Mamba-2, and diffusion.
The original AdamW objectives remain the canonical architecture comparison;
objective/data/optimizer changes such as bell-shaped diffusion time sampling,
Muon/Hyperball, Token-Superposition, curriculum/data selection, and native FP8
must pass their separately bounded research gates before promotion.

No dependency was installed, no dataset or model was downloaded, and no long
GPU job was started during this research/plan update. That research led to the
current Milestone 6A implementation phase above.

## Current update (August 4, 2026, Milestone 7 planning)

Milestone 6 was committed and pushed as
`a171ce7e11c62ba528c1ec3b276c18f3aba46890`. Milestone 7's first offline phase
is now implemented locally: all three pretraining commands expose a seed that
binds both initialization/RNG and shard order, dry-run plans report the seed
and retained-checkpoint storage upper bound, and `plan-architecture-study`
emits a canonical nine-lineage contract under ignored `artifacts/`.

The real plan fixes seeds 1337/2027/3407 and full-scheduler screening stops at
12,208 dense steps (100,007,936 tokens), 14,784 Mamba-2 steps (121,110,528
tokens), and 12,915 diffusion steps (105,799,680 tokens). Only seed 1337 is
marked to resume to the full bounds of 61,036, 73,919, and 64,574 steps. Every
analytic FLOP ratio is within 0.008% of its nominal dense reference, well inside
the predeclared 3% policy. The plan's four-checkpoint-per-lineage retention
upper bound totals about 8.65 GB.

Do not start the nine screening lineages yet. First commit this planner phase,
then run a separately approved synchronized 50-100-step calibration for each
architecture from the clean revision. Feed those measured throughputs back
into the planner to produce wall-time estimates and a final storage/free-space
preflight before requesting approval for the long study.

The post-planner complete gate passes formatting, lint, strict typing, CLI and
lock discovery, and all 180 tests with 85.59% branch coverage.

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
