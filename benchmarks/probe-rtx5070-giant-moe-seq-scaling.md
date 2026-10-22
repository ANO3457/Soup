# Probe (DRAFT): does a longer step raise tokens/s of the synthetic 1T MoE step? RTX 5070 Laptop 8 GB

Status: **DRAFT, work in progress. NOT pre-registered.** No decision rule is fixed in this
version and nothing has been run. This file is a checkpoint of the code reading done so
far; the pre-registered rule replaces it in a later commit, before any run.

Every number this probe will produce is "random weights, shape only; routing forced
uniform", from a measurement harness (`benchmarks/harness/moe_stream_step.py`), not from
`soup train`. Prior art for streaming layers through a small device memory is L2L
(Pudipeddi et al. 2020, "Training Large Neural Networks with Constant Memory using a New
Execution Algorithm").

## Findings so far (code reading, no run)

Base record: `benchmarks/probe-rtx5070-giant-moe-step.md` (1,026,408,232,448 parameters,
K2 shape, 61 layers, 60 MoE layers of 384 experts, 533 GB NF4 store, `--tier disk
--no-pin --seq 512 --steps 2 --route uniform`: 867.95 s and 1,108.77 s per step, peak
allocated VRAM 4,723 MiB, peak working set 7.89 GB).

- Harness knobs: `--seq`, `--batch` (already exists, default 1), `--steps`, `--route
  model|uniform`, `--tier ram|disk`, `--no-pin`, `--buffers`, `--lora-r`. Timed: forward +
  backward + AdamW step, between two `cuda.synchronize()` calls; tokens/s is
  `batch * seq / seconds`.
- Expert reads: `StreamedExperts` (`src/soup_cli/utils/expert_stream.py`) fetches every
  routed-to expert once in forward, once in the checkpoint's recompute and once in its own
  backward, synchronously (read, dequantise, compute, next expert).
- `--route uniform` sends flattened row t to experts 8t .. 8t+7 (mod 384), so every
  expert of every MoE layer is hit once `batch * seq >= 48`. Reads per step are then
  60 x 384 x 3 = 69,120 and ~1.57 TB at every seq of this probe. Tokens per expert per
  pass = `batch * seq / 48`: 10.7 at 512, 21.3 at 1024, 42.7 at 2048, 85.3 at 4096.
- Uniformly random top-8 routing would leave a fraction (376/384)^T of experts untouched:
  2.1e-5 at T = 512 and below 1e-9 from T = 1024, so coverage does not grow with seq
  there either.
- The model's own routing with random weights collapsed onto ~9.7 experts per layer and
  pass at 512 tokens (base record section 4.1); it is not a scaling arm.

## Still to do before this becomes the pre-registered rule

- GPU and RAM peak estimate per seq (activations, checkpoint boundaries, MLA attention and
  the SDPA kernel it gets, logits and loss at vocab 163,840).
- Arms, order, page-cache control, primary metric, thresholds, validity gates, budget.
