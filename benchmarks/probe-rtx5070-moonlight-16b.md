# Probe: LoRA on Moonlight-16B-A3B (DeepSeek-V3 architecture) with streaming, RTX 5070 Laptop 8 GB

Status: RULE COMMITTED, NOT RUN.

`moonshotai/Moonlight-16B-A3B`: `model_type: deepseek_v3` (checked in its
`config.json` before anything was built), 27 layers, 64 routed experts + 2 shared
per MoE layer, top-6, 15.96 B parameters, 31.92 GB of bf16 safetensors. It is
the same decoder Kimi K2 uses, at 1/64 of the size. It does not fit the 8 GB
card resident, at bf16 or at NF4 (~8.5 GB of weights alone).

## 1. Question

Does a LoRA training step of Moonlight-16B run on this laptop under layer
streaming (#1914, routed experts fused and at bf16) and under expert streaming
(the per-expert NF4 layout)? What are its seconds per step, tokens per second,
peak VRAM and peak host memory in each mode?

## 2. Decision rule (written before any run)

Harness: `benchmarks/harness/moe_stream_step.py`, real weights, random token ids,
`--seq 512 --batch 1 --steps 5 --lora-r 8`, LoRA on the attention projections
(Moonlight has `q_lora_rank: null`, so `q_proj`, `kv_a_proj_with_mqa`,
`kv_b_proj`, `o_proj`). Two stores, sharded once beforehand (the sharding time is
reported, not judged):

- arm **L**: `--expert-layout fused`, `--tier disk`. Attention, shared and dense MLP are NF4; the routed experts stay bf16 in the layer files.
- arm **E**: `--expert-layout per_expert`, `--tier ram`. Every expert matrix is NF4, read one expert at a time.

Order: L, E, then E, L (two rounds, interleaved). Step 0 of every run is warm-up
and is reported but not used.

- An arm **COMPLETES** iff in both rounds:
  - all 5 steps finish with a finite loss and finite adapter gradients;
  - peak allocated VRAM is at most 7,500 MiB (the card has 8,151 MiB; above that, Windows spills to shared memory silently);
  - the process's peak working set stays below 20 GB (the owner's 10 GB-free floor on a 31.7 GB box).
- Reported, with no threshold:
  - the median seconds of steps 1-4 per run, and tokens/s = 512 / that;
  - expert bytes read per step;
  - layer loads;
  - store bytes on disk;
  - shard seconds.

  Speed is not the goal: no number is promised before the run.
- A product-path check follows (outside this rule's numbers): `soup train` with a
  `soup.yaml` on this base, one epoch over a small alpaca file. It **COMPLETES**
  iff it finishes with a finite loss and writes an adapter. Wall time and the
  first/last logged loss are reported.

## 3. Box state to stamp at run time

AC power, clock, free RAM and GPU memory before each run, other sessions' holds,
D: free space, torch / transformers / bitsandbytes versions, the harness commit.

## 4. Results

(empty)

## 5. Verdict

(empty)
