# Gate: layer streaming admits Kimi K2 through transformers' own DeepseekV3Config

Status: **rule committed, not run.** Sections 1-3 are the rule, written and committed before
any gate run. Sections 4-6 are filled from the run. CPU only; no GPU, no K2 weight download.

## 1. Question

Kimi-K2-Base and Kimi-K2-Instruct (moonshotai) publish a `config.json` with
`model_type: "kimi_k2"` and an `auto_map` onto the repository's own
`configuration_deepseek.DeepseekV3Config` and `modeling_deepseek.DeepseekV3ForCausalLM`.
Their weights are DeepSeek-V3's published FP8 block format. Layer streaming is Soup's
implementation of layer-to-layer streaming of a frozen base (L2L, Pudipeddi et al. 2020),
and since #1933 it builds the streamed model from transformers' own class with
`trust_remote_code=False`. A real K2 therefore never reaches the architecture allowlist:
`AutoConfig.from_pretrained(trust_remote_code=False)` raises transformers' "contains custom
code" error first (`trainer/stream_setup.py` and `build_meta_skeleton` in
`utils/layer_stream_runtime.py`).

The change under test loads the raw `config.json` dict into transformers' native
`DeepseekV3Config` (`model_type` rewritten to `deepseek_v3`, `auto_map` dropped), but only
after a guard finds no field whose value would make the native class compute differently
from K2's own modelling code. Every K2 variant that fails the guard is refused by name,
naming the field and its value; Kimi-K2-Thinking (compressed-tensors INT4) is one of them.
transformers does the same mapping itself for the text tower of Kimi K2.5
(`Kimi_K25Config.__post_init__` maps a `kimi_k2` text config onto `deepseek_v3`).

Does the rewritten config carry every K2 field (G1)? Does the streamed native model then train
on exactly the numbers a resident native model on the same weights does (G2)? Does the disk
pre-flight size a real K2 cache from its real index (G3)?

## 2. Decision rules (written before any run)

Every tensor comparison is `torch.equal` (bit-exact). No tolerance is substituted after the
fact; a failing check is reported with its maximum absolute difference.

Inputs, read as JSON data only; no K2 Python file is imported or executed anywhere in this
gate. The files were fetched (text only) into the session scratchpad:

| file | revision | sha256 |
|---|---|---|
| Kimi-K2-Base `config.json` | `ce72df012259dcc55d945e890f815fe7ef69159c` | `8c13ae1049df55f29b3bdcae69a562433f243ff70dac251d819ecad8dbdf7439` |
| Kimi-K2-Instruct `config.json` | `fd1984e2b7a3350dbf7305fe73a4ede25c14de50` | same bytes as Base |
| Kimi-K2-Thinking `config.json` | `a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55` | `666ee58fc3871ecc369a22bc8c100075c2ad24ccbe8b3e336fa5f726e6111c15` |
| Kimi-K2-Base `model.safetensors.index.json` | as Base | `c1f1d16c853f20467ae81361d2a92223650d39efa005f9c872a7cc14425ddcbc` |
| `modeling_deepseek.py` (all three) | as above | `ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4` |

### G1, config (CPU, no model)

**G1a, mapping.** Soup's loader (`load_stream_model_config`) is run on a temporary folder
holding the real Base `config.json`, and again on one holding the Instruct file. It must
return a `DeepseekV3Config` whose `model_type` is `deepseek_v3` and that has no `auto_map`.
For every key of the raw JSON except `model_type` and `auto_map`, the loaded config must hold
the same value: `getattr(config, key) == raw[key]`. Two transformers normalisations are
named here in advance; an exploratory probe during design (a scratch script, before this
rule was written) showed that transformers applies them to any `config.json`, and they are
the only ones admitted:

- `rope_theta` is read as `config.rope_parameters["rope_theta"]`;
- `torch_dtype` is compared as `str(config.torch_dtype) == "torch." + raw["torch_dtype"]`.

Also: every key of the raw `rope_scaling` equals `config.rope_parameters[key]`, and
`config.rope_parameters["rope_type"] == "yarn"`.

Controls (each must hold, or G1a pins nothing):

- today's failure: `AutoConfig.from_pretrained(folder, trust_remote_code=False)` on the same
  folder raises `ValueError`;
- AutoConfig equivalence: the same dict with `model_type: deepseek_v3` and no `auto_map`,
  written to a second folder and loaded by `AutoConfig.from_pretrained`, gives a `to_dict()`
  equal to the K2 path's `to_dict()` except `_name_or_path`;
- the loader never calls transformers' dynamic-module loader
  (`get_class_from_dynamic_module` replaced by a function that fails the test), and the raw
  dict it reads is not mutated.

Reported, no threshold: every warning or log record at WARNING level or above emitted
during the load.

**G1b, the guard.** The guard checks, in this order, and a failure names the field and its
value:

| field | admitted | why the native class would compute differently otherwise |
|---|---|---|
| `architectures` | exactly `["DeepseekV3ForCausalLM"]` | another head is not the decoder this mapping was checked on |
| `topk_method` | `"noaux_tc"` | K2's router implements only this; the native router always runs it |
| `scoring_func` | `"sigmoid"` | K2's router implements only this; the native router always uses it |
| `n_group` | `1` | K2 masks unselected groups with 0.0, the native router with -inf; with one group nothing is masked |
| `topk_group` | `1` | as `n_group` |
| `num_experts_per_tok` | an int >= 2 when `norm_topk_prob` is true or absent | K2 renormalises the top-k weights only when top-k > 1; the native router always does |
| `moe_layer_freq` | absent or 1 | K2 makes only every n-th layer an MoE layer; the native class makes every layer past `first_k_dense_replace` one |
| `ep_size` | absent or an int <= 1 | K2 splits the experts across processes |
| `n_routed_experts` | an int >= 1 | `None` makes K2 all-dense; the native class cannot build it |
| `n_shared_experts` | an int >= 0 | `None` drops K2's shared expert; the native class always builds one |
| `num_key_value_heads` | absent or equal to `num_attention_heads` | K2's MLA ignores it; the native class divides by it |
| `rope_interleave` | absent or true | K2 always applies the interleaved rotary layout; the native class reads this flag |
| `rope_scaling` | absent, or a yarn dict as below | |
| `quantization_config` | absent, or a spelling `utils/fp8_block.py`'s `fp8_block_size` accepts | Soup dequantises only DeepSeek-V3's block FP8 |

A `rope_scaling` dict is admitted only when: its `type` is `"yarn"` (and `rope_type`, if
present, is too); it has no key outside `type`, `rope_type`, `factor`,
`original_max_position_embeddings`, `beta_fast`, `beta_slow`, `mscale`, `mscale_all_dim` (the
keys K2's yarn reads; the native yarn reads more, such as `attention_factor` and `truncate`);
`factor` is a number; `original_max_position_embeddings` is present (K2 defaults it to 4096,
the native class to `max_position_embeddings`); `beta_fast` and `beta_slow` are not 0 where
present (the native yarn replaces a 0 with its default); and `mscale` is consistent with
`mscale_all_dim` (if `mscale_all_dim` is non-zero, `mscale` must be non-zero; otherwise
`mscale` must be absent or 1), the cases where the native attention factor equals K2's.

- **PASS (guard)**: the unmodified Base and Instruct dicts pass. For every row above (and for
  each `rope_scaling` condition), a copy of the real Base dict with exactly that one field
  changed to a failing value is refused with `UnsupportedKimiK2Error`, whose message names the
  field and the value. The FP8 spellings `fp8_block_size` accepts (with `scale_fmt: "float"`,
  without `fmt`) pass. Thinking is refused; the record names the first failing field
  (expected, by reading: `quantization_config`, `compressed-tensors`). A refusal never prints a
  raw control byte from the config.

**G1c, the guard is necessary on the numeric rows (reference transcribed by hand).** K2's
remote code does not import on transformers 5.17 (the investigation), and this gate does not
try to run it. Instead the router (`MoEGate.forward`) and the yarn rotary embedding plus its
softmax scale are transcribed from `modeling_deepseek.py`, read as text, into the test file,
and compared with the native `DeepseekV3TopkRouter`, the native rotary embedding and the
native attention's `scaling`:

- equal (`torch.equal` / `==`) on K2's own values and on every admitted value tested;
- different on each refused variant tested: `n_group` 2 with a negative router bias,
  `num_experts_per_tok` 1 with `norm_topk_prob`, no `original_max_position_embeddings`,
  `mscale_all_dim` without `mscale`, `mscale` != 1 without `mscale_all_dim`, a 0 `beta_fast`,
  and an `attention_factor` key. A transcription that raises where the native module computes
  counts as different (by reading, a 0 `beta_fast` divides by zero in K2's yarn).

A transcription can be wrong; this is evidence for the guard, not vendor-code parity.

### G2, streamed-vs-resident parity on a tiny K2-like checkpoint (CPU fp32)

Model: a seeded random native `DeepseekV3ForCausalLM`: vocab 128, hidden 64, 3 layers with
`first_k_dense_replace=1`, 8 routed experts plus a shared expert, top-k 2, `n_group=1`,
`topk_group=1`, `q_lora_rank` 24, `kv_lora_rank` 16, `qk_nope_head_dim` 16,
`qk_rope_head_dim` 8, `v_head_dim` 16, 4 heads, yarn rope (`factor` 32,
`original_max_position_embeddings` 64, `max_position_embeddings` 2048, `beta_fast` 1,
`beta_slow` 1, `mscale` 1, `mscale_all_dim` 1, `rope_theta` 50000), `routed_scaling_factor`
2.827, and `e_score_correction_bias` set to large random values. Weights saved bf16 (the
bias float32). `config.json` is the real Base dict with these dimensions substituted: it
keeps `model_type: kimi_k2`, `auto_map`, `architectures`, `topk_method`, `scoring_func`,
`moe_layer_freq`, `seq_aux`, `aux_loss_alpha` and `num_nextn_predict_layers: 0`. Every layer
also carries a stale BF16 `model.layers.N.self_attn.rotary_emb.inv_freq` (shape [7], filled
with a constant no rope uses), as the real checkpoint carries 61 of them. The folder holds no
`.py` file. The FP8 copy is written by `benchmarks/harness/fp8_block_quantize.py` (block
16 x 32) from the bf16 checkpoint.

Resident control: `DeepseekV3ForCausalLM` from a `DeepseekV3Config` built in the test from
keyword arguments (the native `rope_parameters` spelling, not through Soup's loader), fp32,
`load_state_dict(strict=True)` from the checkpoint's tensors minus the stale `inv_freq` keys
(the dropped set must be exactly those keys); for the FP8 cells, from the test's independent
block-loop dequantisation of the FP8 copy. `per_expert` cells set the resident experts
implementation to `eager`, as the FP8 gate does. LoRA: the targets Soup resolves for the
loaded config under `target_modules: auto`, rank 4, the same seeded adapters on both sides.

Cells, CPU, `dtype=float32`, `quantization: none`: {fused, per_expert} x {bf16 source, FP8
block source}.

- **PASS (model)**: in every cell the streamed model's logits, loss and every LoRA gradient
  equal the resident control's; both have the same set of LoRA gradients, and at least one
  layer-0 LoRA gradient is non-zero.
- **PASS (stale key)**: the runtime streams no `rotary_emb.inv_freq` key into any layer, and
  the streamed model has no per-layer `rotary_emb` module. Whether the sharder writes the key
  into the layer files is reported.
- **Controls** (each must hold, or the cells pin nothing): the resident model with the
  default rope instead of yarn gives different logits; zeroing the router bias changes the
  logits; the FP8-dequantised resident differs from the bf16 resident; the streamed model's
  config is a `DeepseekV3Config` with `model_type` `deepseek_v3`, no `auto_map` and (FP8
  cells) no `quantization_config`; no dynamic-module load happens.
- **FAIL**: any check false.

### G3, the real index (CPU, headers only, nothing downloaded)

No K2 shard is local, so the headers are synthesised: names from the real Base index,
shapes from the real config, and dtypes from the header facts the investigation recorded:
F8_E4M3 for every `*.weight` that has a `*.weight_scale_inv` beside it; F32 for those scales
(shape `[ceil(rows/128), ceil(cols/128)]`) and for `mlp.gate.e_score_correction_bias`; BF16
for the norms, the embeddings, `lm_head`, the router weight and the stale
`rotary_emb.inv_freq`. The stale key's shape is [56]: during design, a probe showed that it
is the shape for which the map's byte total equals the index's `total_size` exactly ([32],
`qk_rope_head_dim / 2`, leaves 2,928 B unaccounted for).

- **G3a, PASS**: the synthetic map's key set equals the index's key set (139,644 keys), and
  its byte total equals the index's `metadata.total_size`, 1,029,173,256,720 B.
- **G3b**: Soup's `estimate_shard_cache_bytes` runs over the map (only `read_header` is
  replaced, to serve it; every other step is Soup's), with `dtype=bfloat16`, the arch
  `stream_arch_of` returns for the config Soup's loader builds from the real `config.json`,
  `quant=nf4`, double quantisation on, the quant suffixes of
  `quantised_layer_suffixes(build_meta_skeleton(<folder with the real config>, quant="nf4"))`,
  `num_hidden_layers` 61, the config's `quantization_config`, in both layouts.
  **PASS**: no refusal, and in both layouts every layer's payload and the payload outside the
  layers equal an oracle computed in the gate script from the shapes alone (NF4 with double
  quantisation per weight of `n` elements: `ceil(n/2) + ceil(n/64) + 4*ceil(ceil(n/64)/256) + 4`
  bytes; everything else at its stored dtype).
- Reported: the total (payload plus the header and index allowances) and the RAM-tier bytes
  for each layout, compared with the investigation's figures (per_expert about
  533,246,623,322 B, fused about 2,039,152,149,798 B). An exact match is reported as one; a
  difference is reported with its size and explained, not adjusted, and is not a FAIL by
  itself.

### Mutation checks

Each guard row, each `rope_scaling` condition and each rewrite step (`model_type`, `auto_map`)
is altered in the source, a named test must fail, and the file is restored byte for byte
(sha256 checked). A mutation no test catches is reported and the test strengthened.

## 3. Not shown here

- **Parity with K2's own code.** Not run: the remote code does not import on transformers
  5.17, and Soup never executes it. G1c compares against a hand transcription only.
- **Known numeric differences from K2's own code**, not gated: the MoE combine (K2 sums the
  weighted expert outputs in fp32 and rounds once; the native class rounds each weighted
  output to the model dtype and accumulates with `index_add_`, bf16 rounding noise), eager
  versus sdpa attention, and the router bias held as a Parameter (K2) versus a float32
  buffer (native; both float32 in the checkpoint).
- **The tokenizer.** K2's tokenizer exists only as remote code and needs `tiktoken`; it is
  not exercised here.
- No GPU, no NF4 model parity (those tests run only on CUDA), no real K2 weights, no K2 or 1T
  training run, no throughput.

## 4. Box state

(filled after the run)

## 5. Results

(filled after the run)

## 6. Mutation run

(filled after the run)
