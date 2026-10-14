# Probe: one LoRA training step on a Kimi K2-shaped store, RTX 5070 Laptop 8 GB

Status: RULE COMMITTED, NOT RUN.

**Every number in this record is "random weights, shape only".** The store is
synthetic: the parameter count and tensor shapes of `moonshotai/Kimi-K2-Instruct`
(~1.03 T parameters, 61 layers, 384 routed experts of 3 x 7168 x 2048 per MoE
layer), random NF4 bytes. It says nothing about the quality, convergence or
throughput of training the real model. Prior art for streaming layers through a
small device memory is L2L (Pudipeddi et al. 2020, "Training Large Neural
Networks with Constant Memory using a New Execution Algorithm"); this record
does not use L2L's layer-major micro-batching.

## 1. Question

With layer streaming plus expert-granularity streaming (`expert_layout="per_expert"`,
`benchmarks/gate-expert-streaming.md`), does ONE full LoRA training step (forward,
backward, AdamW on attention adapters) of a K2-shaped model complete on this
laptop, and what does it cost in time, VRAM, host RAM and bytes read?

## 2. Decision rule (written before any run)

Store: `benchmarks/harness/synth_moe_store.py --shape k2 --out D:/soup-bench/moe-k2-synth`.
Step: `benchmarks/harness/moe_stream_step.py --seq 512 --batch 1 --steps 2 --lora-r 8`
(step 0 is cold, step 1 is reported separately; neither is averaged into the other).

- **COMPLETES** iff step 0 and step 1 both finish with a finite loss and finite
  adapter gradients, the peak allocated VRAM is at most 7,500 MiB (the card has
  8,151 MiB; the margin is for the CUDA context, and Windows would otherwise spill
  to shared memory silently), and the process's peak working set stays below
  20 GB (the box has 31.7 GB; the rule keeps the owner's 10 GB-free floor).
- **DOES NOT COMPLETE** otherwise; the record then states which limit, at which
  layer, with the error text.
- Reported, no threshold: seconds per step, tokens per second, expert bytes read
  per step against the 3 x routed-expert bytes the design predicts (forward,
  checkpoint recompute, backward), layer loads.

Estimate before the run (an estimate, not a measurement): ~570 GB of experts read
3 times per step, ~1.7 TB. The async disk tier reads this drive at 4.09-4.14 GB/s
(`probe-rtx5070-what-bounds-streaming.md` §21); the expert reader is synchronous
(one expert, then its compute), so expect it slower: 7 minutes at 4.1 GB/s, about
24 at 1.2 GB/s.

## 3. Box state to stamp at run time

AC power, clock, free RAM before the run, GPU memory used before the run, other
sessions' holds, D: free space, torch / transformers / bitsandbytes versions,
the commit the harness ran from, the store's `synth.json`.

## 4. Results

(empty)

## 5. Verdict

(empty)
