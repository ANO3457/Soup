# Gate: expert-granularity streaming is bit-exact against the resident MoE block

Status: RULE COMMITTED, NOT RUN. Everything below "Results" is empty until the run.

## 1. Question

`utils/expert_stream.py` replaces a MoE layer's routed-experts module with
`StreamedExperts`, which reads one expert at a time from a per-layer experts file
and fetches it again in the backward. Does it give the same numbers as the
resident experts module on a REAL MoE checkpoint, not only on the random tiny
models of `tests/test_expert_stream.py`?

## 2. Decision rule (written before any run)

Model: `allenai/OLMoE-1B-7B-0924` from the local Hugging Face cache (64 experts,
top-8, 16 layers; already on disk since the #992 coverage probe). Layers 0, 7
and 15. For each layer:

- the resident control is transformers' `OlmoeExperts` holding that layer's real
  expert weights, run with `experts_implementation="eager"`;
- the streamed arm is `StreamedExperts` over an experts file written by
  `write_expert_shard` from the same tensors;
- the routing is the layer's real router (`mlp.gate.weight`) applied to the
  input: softmax then top-8, as `OlmoeTopKRouter` does;
- the input is 512 tokens of N(0, 1) hidden states (seed 1234), scaled by the
  layer's `post_attention_layernorm` weight (the norm the MoE block reads), and
  an N(0, 1) output gradient (seed 4321).

Harness: `benchmarks/harness/expert_stream_gate.py`; raw JSON under
`benchmarks/results/probe-rtx5070/expert-stream/`.

Cells: (a) CPU fp32, unquantised; (b) CUDA bf16, unquantised; (c) CUDA bf16 with
every expert matrix NF4-quantised by the sharder, the control holding the
dequantised weights. Each cell compares, with `torch.equal`: the output, the
gradient of the input, and the gradient of the routing weights.

- **PASS**: `torch.equal` holds for all 3 layers x 3 cells x 3 tensors (27 checks),
  and the streamed arm read each routed-to expert exactly twice (forward +
  backward) and no other expert.
- **FAIL**: any check false. The record then states the max absolute difference
  per failing tensor; no tolerance is substituted after the fact.

Not measured here: speed (expert streaming promises none; the #992 probe found a
step routes to 96.7-100% of experts), a whole-model step, any DeepSeek-V3 weights.

## 3. Box state to stamp at run time

AC power, clock, free RAM, GPU memory used before the run, foreign readers
(other sessions' holds in `.claude/SESSIONS.md`), torch / transformers /
bitsandbytes versions, the commit the harness ran from.

## 4. Results

(empty)

## 5. Verdict

(empty)
