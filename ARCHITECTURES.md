# Architecture study

Milestone 7 is complete at revision `53d071b`. The study uses the frozen
plan in `artifacts/architecture-study/plan.json`, the TinyStories 16K shard
manifest, sequence length 1024, microbatch 8, one accumulation step, and
seeds 1337, 2027, and 3407. Every lineage used its full 500M-dense-equivalent
scheduler from step zero. Only seed 1337 continued from screening to the full
budget. Diffusion used fused AdamW, the only optimizer promoted by Milestone
6A; dense and Mamba-2 used the compiled-default/auto backend.

The machine was an NVIDIA RTX 4080 SUPER under WSL2. The canonical result
report is [`reports/zero-20m-architecture-study.json`](reports/zero-20m-architecture-study.json).
Its audit validated all 15 retained checkpoints, including model and recovery
hashes. The ignored event logs and checkpoints remain under
`artifacts/architecture-study/`.

## Terminal training results

The `loss` column is the final per-step training objective. Diffusion loss is
masked-reconstruction loss and is not directly comparable to the causal-LM
losses. Throughput is the median of per-step telemetry after the first 100
optimizer steps; it includes the current metric/telemetry path and is not a
hardware-only benchmark.

| Architecture | Seed | Stage | Optimizer step | Tokens | Final loss | Median tokens/s |
|---|---:|---|---:|---:|---:|---:|
| Dense OLMo2 | 1337 | full | 61,036 | 500,006,912 | 1.2459 | 398,437 |
| Dense OLMo2 | 2027 | screening | 12,208 | 100,007,936 | 1.6404 | 404,568 |
| Dense OLMo2 | 3407 | screening | 12,208 | 100,007,936 | 1.8980 | 405,558 |
| Mamba-2 | 1337 | full | 73,919 | 605,544,448 | 1.2530 | 170,333 |
| Mamba-2 | 2027 | screening | 14,784 | 121,110,528 | 1.6381 | 170,663 |
| Mamba-2 | 3407 | screening | 14,784 | 121,110,528 | 1.7458 | 170,896 |
| Masked diffusion | 1337 | full | 64,574 | 528,990,208 | 3.5997 | 418,766 |
| Masked diffusion | 2027 | screening | 12,915 | 105,799,680 | 2.7481 | 417,237 |
| Masked diffusion | 3407 | screening | 12,915 | 105,799,680 | 4.0696 | 415,818 |

These terminal losses are descriptive checkpoints, not the final architecture
ranking. The planned comparison still requires architecture-appropriate
validation and downstream evaluation; causal perplexity must not be assigned
to the masked-diffusion objective.

## Reproducibility and next use

- The three dense screening checkpoints are the reusable Milestone 8 dense
  baseline only when Milestone 8 preserves the identical tokenizer, shard,
  scheduler prefix, seed, and configuration hashes.
- The seed-1337 full checkpoints are the required full architecture-study
  continuations and must not be retrained under a duplicate label.
- The first dense seed-1337 wrapper timeout is retained separately as
  `artifacts/architecture-study/dense/seed-1337-timeout-20260805` and is
  excluded from all canonical counts.
- Next: run the architecture-specific validation/evaluation pass, then begin
  Milestone 8 controlled dense ablations using the reusable screening baseline.
