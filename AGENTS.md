# Repository instructions

## Scope

These instructions apply to the entire `lm-from-zero` repository. Follow the
workspace-level instructions as well; this file is authoritative for
project-specific commands and conventions.

## Project purpose

Build language models from first principles in the milestone order documented
in `plans/01-lm-from-zero.md`. Keep the dense OLMo2-compatible vertical slice
as the production path. Mamba-2 and masked diffusion remain later,
compute-matched research tracks.

## Setup

- Primary environment: the `Ubuntu-24.04` WSL2 distro with Python 3.12. The
  separate distro named `Ubuntu` does not currently provide `uv`.
- Keep the main environment at `.venv` and any future vLLM environment at
  `.venv-vllm`.
- Use `uv` for environment creation, dependency locking, and command execution.
- Do not install or update dependencies, create environments, download data,
  create caches, or change host tooling without explicit user confirmation.
- Never install Python packages globally.
- Keep the default development and test path CPU-only and offline.

Create and synchronize the approved environment with:

```bash
uv venv --python 3.12 .venv
uv sync --frozen --link-mode copy
```

The optional pinned Mamba-2 Triton oracle is source-only for the current
PyTorch release. Install its independent Triton SSD path without attempting the
unrelated `selective_scan_cuda` extension:

```bash
MAMBA_SKIP_CUDA_BUILD=TRUE \
uv sync --frozen --group mamba-oracle --link-mode copy
```

`--link-mode copy` avoids cross-filesystem hard-link warnings between the WSL
cache and the repository on the Windows-mounted filesystem.

## Commands

Full verification:

```bash
PYTHONPATH=src uv run --frozen ruff format --check .
PYTHONPATH=src uv run --frozen ruff check .
PYTHONPATH=src uv run --frozen mypy src tests
PYTHONPATH=src uv run --frozen pytest
PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli --help
```

When the ignored TinyStories sample exists, verify it separately with
`PYTHONPATH=src uv run --frozen python -m lm_from_zero.cli verify-sample
data/tinystories/manifest.json`.

Tokenizer training, oracle verification, and benchmark commands are documented
in `README.md`. Mamba oracle verification requires the separately synchronized
`mamba-oracle` group and a CUDA GPU, but not a host CUDA toolkit. The same
document contains the deterministic shard build and full-build verification
commands, the dense-model configuration summary, and the focused checkpoint,
runner, evaluation, Hugging Face export, and native generation test commands.
The pretraining command prints a dry-run plan unless `--execute` is explicitly
supplied.

Interactive long-running commands use the pinned `tqdm` dependency for live
phase/step progress on stderr. Set `LM_FROM_ZERO_PROGRESS=1` to force the bar
when a wrapper does not provide a TTY, or `LM_FROM_ZERO_PROGRESS=0` to disable
it. Progress output never replaces canonical JSON, JSONL, or Parquet artifacts.
Milestone 6A calibration begins with the CPU-only
`plan-acceleration-calibration` command from a clean committed revision. Its
instrumented per-cell CUDA executor and all measurement artifacts remain
approval-gated; run baselines before candidates and never hand-author
calibration results. A short CUDA smoke validates execution and artifact
integrity only; promotion requires all three fixed-step repetitions to pass
the numerical and throughput-stability gates. Run the focused contract tests
with `PYTHONPATH=src uv run --frozen pytest
tests/test_acceleration_calibration.py
tests/test_acceleration_execution.py tests/test_acceleration_runtime.py
tests/test_acceleration_statistics.py --no-cov`.
Checkpoint evaluation must use fixed non-wrapping shard windows. Checkpoints
use canonical manifests, separate Safetensors model weights, and
restricted-load recovery state under ignored `artifacts/`. Do not copy measured
values manually into reports. Training JSONL is the durable metric source;
TensorBoard and atomic Parquet outputs are rank-zero mirrors that must remain
rebuildable from canonical JSONL. The buffered JSONL sink must durably sync
before every checkpoint and on snapshot, abort, or clean shutdown. Long-run
checkpoint cadence is time-only by default; use explicit step cadence only for
bounded tests that require it.

Dependency-free fallback verification:

```bash
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
python3.12 -m compileall -q src tests
```

Windows CPU fallback:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m unittest discover -s tests -v
py -3.12 -m compileall -q src tests
```

## Conventions

- Support Python 3.12 and use complete type annotations on public APIs.
- Keep core implementations project-owned. Do not import `transformers` in
  tokenizer training, core models, pretraining, or project-owned post-training
  losses. Transformers is allowed only in export, parity, and adapter code.
- Prefer small architecture-specific modules behind explicit shared protocols.
- Make determinism visible: stable tie-breaks, explicit seeds, canonical JSON,
  content hashes, and tested resume state.
- Reject invalid configuration and incompatible artifacts before expensive
  allocation or training.
- Use atomic writes for checkpoints, manifests, tokenizer models, and other
  stateful artifacts.
- Validate every checkpoint artifact and recovery binding before mutating a
  live model, optimizer, scaler, or RNG state. Load recovery payloads only
  through the restricted `torch.load(weights_only=True)` path.
- Keep pretraining dry-run-first. A short smoke and fresh explicit approval are
  required before adding `--execute` for a long run.
- Keep GPU, network, hosted-service, and publication paths optional.
- Add tests with behavior. Include negative-path and corruption tests where
  serialized state or training recovery is involved.
- Do not type measured results into reports; generate them from recorded data.

## Data and artifact policy

- Raw datasets, token shards, caches, checkpoints, exports, GGUF files, logs,
  and local environments stay outside Git.
- Never commit secrets, account data, private prompts, personal paths, or
  identifying metadata.
- Record public dataset provenance, immutable revisions, licenses, hashes, and
  filtering decisions in machine-readable manifests.

## Verification

Before declaring a change complete, run the narrow tests for the edited code
and the full offline CPU suite. When the configured tools exist, also run Ruff,
strict mypy, pytest, and coverage. Report any command that could not run and
why.

GPU work must begin with a short smoke test. Long training, downloads, hosted
compute, external login, publishing, Git pushes, releases, and pull requests
require explicit user confirmation.

The offline runner suite includes a real two-process CPU/Gloo DDP test. DDP
commands must be launched through `torchrun`; rank zero alone owns logs,
checkpoint publication, and retention. Keep global token accounting and
rank-local cursor/RNG recovery covered when changing the runner.

The `prepare-dpo-holdout` command is CPU-only and builds a deterministic
preference split excluded from the recorded DPO training mix. The
`evaluate-dpo-holdout` command performs CUDA scoring of both final checkpoints;
validate the held-out split and checkpoint bindings first, then obtain fresh
approval before running its `--max-pairs 8` smoke. A separate approval is
required before the unbounded full evaluation.
