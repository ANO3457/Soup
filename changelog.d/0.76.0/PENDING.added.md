- **Layer streaming admits Kimi K2 (`model_type: kimi_k2`).** Kimi-K2-Base and Kimi-K2-Instruct are
  DeepSeek-V3's decoder in DeepSeek-V3's FP8 block format, but their `config.json` declares
  `model_type: kimi_k2` with an `auto_map` onto the repository's own code, so `AutoConfig` refused
  them as custom code before the architecture list was reached (layer streaming never runs a
  repository's model code). The streamed model's config is now read from that file as plain data
  into transformers' own `DeepseekV3Config` (`model_type` set to `deepseek_v3`, `auto_map` dropped),
  after a guard that admits a K2 config only where that class computes what K2's own code computes:
  `noaux_tc` sigmoid routing with one expert group, top-k of at least 2 under renormalisation, every
  layer past `first_k_dense_replace` an MoE layer, no expert parallelism, only the yarn rope keys
  K2 reads, and unquantised or DeepSeek-V3 block FP8 weights. A config that fails any of them is
  refused before any weight is resolved, naming every such field and its value; Kimi-K2-Thinking
  (compressed-tensors INT4) is refused as `quantization_config.quant_method='compressed-tensors'`.
  In the gate (`benchmarks/gate-kimi-k2-admission.md`) the field-by-field config check fails as
  registered on one field: transformers adds `rope_theta` and `rope_type` to `rope_scaling`, as it
  does for any DeepSeek-V3 config. The other checks pass. A tiny K2-like checkpoint (the published
  config with tiny dimensions) streams bit for bit like a resident `DeepseekV3ForCausalLM` on the
  same weights (logits, loss, LoRA gradients; `fused` and `per_expert`; bf16 and FP8 block), and the
  disk pre-flight sizes the published Base layout exactly:
  533.2 GB per_expert and 2.04 TB fused under `quantization: 4bit`
  (`benchmarks/gate-kimi-k2-admission.md`; CPU only, no K2 weights downloaded, nothing trained on
  K2). K2's tokenizer is repository code: it needs `--trust-remote-code`, which reaches the tokenizer
  only, and `pip install tiktoken`, which Soup does not install; a failed tokenizer load on a K2
  base now says so. Known differences from K2's own code remain: the MoE combine's rounding and
  eager versus sdpa attention. (#PENDING)
