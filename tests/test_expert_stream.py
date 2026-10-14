"""Expert-granularity streaming (``utils/expert_stream.py``).

The gate is bit-exactness against transformers' eager experts loop: the
streamed module must equal the resident one forward AND backward, on the
block alone and inside a streamed DeepSeek-V3 model.
"""

from __future__ import annotations

import pytest

CUDA = pytest.mark.gpu(reason="NF4 dequantisation needs bitsandbytes on CUDA")


# --------------------------------------------------------------------------
# block level
# --------------------------------------------------------------------------
def _deepseek_v3_experts(seed):
    import torch
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts

    config = DeepseekV3Config(
        hidden_size=32, moe_intermediate_size=16, n_routed_experts=8, n_group=2,
        topk_group=1, num_experts_per_tok=2,
    )
    config._experts_implementation = "eager"
    torch.manual_seed(seed)
    module = DeepseekV3Experts(config)
    with torch.no_grad():
        module.gate_up_proj.normal_(0.0, 0.2)
        module.down_proj.normal_(0.0, 0.2)
    module.requires_grad_(False)
    return module, 8, 2


def _olmoe_experts(seed):
    import torch
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    config = OlmoeConfig(hidden_size=32, intermediate_size=16, num_experts=8, num_experts_per_tok=2)
    config._experts_implementation = "eager"
    torch.manual_seed(seed)
    module = OlmoeExperts(config)
    with torch.no_grad():
        module.gate_up_proj.normal_(0.0, 0.2)
        module.down_proj.normal_(0.0, 0.2)
    module.requires_grad_(False)
    return module, 8, 2


def _streamed_copy(tmp_path, module, n_experts):
    from soup_cli.utils.expert_stream import (
        ExpertFetcher,
        ExpertShardReader,
        streamed_experts_class,
        write_expert_shard,
    )

    path = str(tmp_path / "experts_001.safetensors")
    pairs = (
        (module.gate_up_proj[e].detach(), module.down_proj[e].detach())
        for e in range(n_experts)
    )
    info = write_expert_shard(path, pairs, quantise=False)
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs={}, dtype="float32"
    )
    streamed = streamed_experts_class()(
        reader, ExpertFetcher(device="cpu", dtype="float32"), module.act_fn, 1
    )
    return streamed, reader


def _routing(n_tokens, n_experts, top_k, hidden, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand(n_tokens, n_experts, generator=generator)
    top_k_index = scores.topk(top_k, dim=-1).indices
    top_k_weights = torch.rand(n_tokens, top_k, generator=generator).requires_grad_(True)
    states = torch.randn(n_tokens, hidden, generator=generator).requires_grad_(True)
    grad_out = torch.randn(n_tokens, hidden, generator=generator)
    return states, top_k_index, top_k_weights, grad_out


@pytest.mark.parametrize(
    "build", [_deepseek_v3_experts, _olmoe_experts], ids=["deepseek_v3", "olmoe"]
)
def test_streamed_experts_forward_equals_eager_bit_exactly(tmp_path, build):
    import torch

    module, n_experts, top_k = build(5)
    streamed, reader = _streamed_copy(tmp_path, module, n_experts)
    states, index, weights, _ = _routing(24, n_experts, top_k, 32, 9)
    with torch.no_grad():
        want = module(states, index, weights)
        got = streamed(states, index, weights)
    assert torch.equal(got, want), (got - want).abs().max().item()
    reader.close()


@pytest.mark.parametrize(
    "build", [_deepseek_v3_experts, _olmoe_experts], ids=["deepseek_v3", "olmoe"]
)
def test_streamed_experts_backward_equals_eager_bit_exactly(tmp_path, build):
    """Input and routing-weight gradients, the two a frozen expert passes on."""
    import torch

    module, n_experts, top_k = build(5)
    streamed, reader = _streamed_copy(tmp_path, module, n_experts)
    states, index, weights, grad_out = _routing(24, n_experts, top_k, 32, 9)

    module(states, index, weights).backward(grad_out)
    want_states, want_weights = states.grad.clone(), weights.grad.clone()
    states.grad = None
    weights.grad = None
    streamed(states, index, weights).backward(grad_out)

    assert float(want_states.abs().sum()) > 0.0
    assert torch.equal(states.grad, want_states), (states.grad - want_states).abs().max().item()
    assert torch.equal(weights.grad, want_weights)
    reader.close()


def test_each_routed_expert_is_read_once_forward_and_once_backward(tmp_path):
    """The cost the docstring states: an expert nobody routed to is never read."""
    import torch

    module, n_experts, _ = _deepseek_v3_experts(5)
    streamed, reader = _streamed_copy(tmp_path, module, n_experts)
    states = torch.randn(6, 32, requires_grad=True)
    index = torch.tensor([[0, 3]] * 6)
    weights = torch.full((6, 2), 0.5, requires_grad=True)
    out = streamed(states, index, weights)
    assert reader.fetches == 2
    out.sum().backward()
    assert reader.fetches == 4
    reader.close()


# --------------------------------------------------------------------------
# experts file + reader
# --------------------------------------------------------------------------
def test_expert_shard_stores_gate_rows_before_up_rows(tmp_path):
    import torch
    from safetensors.torch import load_file

    from soup_cli.utils.expert_stream import write_expert_shard

    gate_up = torch.cat((torch.full((3, 4), 1.0), torch.full((3, 4), 2.0)))
    down = torch.full((4, 3), 3.0)
    path = str(tmp_path / "experts_001.safetensors")
    info = write_expert_shard(path, [(gate_up, down), (gate_up + 10, down + 10)], quantise=False)
    stored = load_file(path)
    assert info.count == 2 and dict(info.shapes) == {"gate_up": (6, 4), "down": (4, 3)}
    assert torch.equal(stored["1.gate_up"][:3], torch.full((3, 4), 11.0))
    assert torch.equal(stored["1.down"], torch.full((4, 3), 13.0))


def test_expert_shard_refuses_an_inconsistent_expert(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import write_expert_shard

    good = (torch.zeros(6, 4), torch.zeros(4, 3))
    bad = (torch.zeros(8, 4), torch.zeros(4, 3))
    with pytest.raises(ValueError, match="expert 1 gate_up has shape"):
        write_expert_shard(str(tmp_path / "x.safetensors"), [good, bad], quantise=False)


def test_reader_refuses_a_file_missing_an_expert(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import ExpertShardReader, write_expert_shard

    path = str(tmp_path / "experts_001.safetensors")
    info = write_expert_shard(path, [(torch.zeros(6, 4), torch.zeros(4, 3))], quantise=False)
    with pytest.raises(ValueError, match="has no tensor '1.gate_up'"):
        ExpertShardReader(path, count=2, shapes=info.shapes, quant_specs={}, dtype="float32")


def test_reader_refuses_a_file_with_more_experts_than_the_index(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import ExpertShardReader, write_expert_shard

    path = str(tmp_path / "experts_001.safetensors")
    pair = (torch.zeros(6, 4), torch.zeros(4, 3))
    info = write_expert_shard(path, [pair, pair], quantise=False)
    with pytest.raises(ValueError, match="holds more than 1 experts"):
        ExpertShardReader(path, count=1, shapes=info.shapes, quant_specs={}, dtype="float32")


# --------------------------------------------------------------------------
# a whole streamed DeepSeek-V3 with per-expert layout vs the resident model
# --------------------------------------------------------------------------
def _tiny_deepseek_v3(tmp_path):
    import torch
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    torch.manual_seed(11)
    config = DeepseekV3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=3, first_k_dense_replace=1, num_attention_heads=4,
        num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8, n_group=2,
        topk_group=1, num_experts_per_tok=2, kv_lora_rank=16, q_lora_rank=24,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, max_position_embeddings=128,
    )
    model = DeepseekV3ForCausalLM(config).to(torch.float32).eval()
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for layer in model.model.layers[1:]:
            bias = layer.mlp.gate.e_score_correction_bias
            bias.copy_(torch.randn(bias.shape, generator=generator))
    weights = tmp_path / "model"
    model.save_pretrained(str(weights))
    # The streamed experts follow the eager loop; the resident control must too.
    model.set_experts_implementation("eager")
    return str(weights), model


def _lora():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_b_proj", "kv_b_proj", "o_proj"], task_type=TaskType.CAUSAL_LM,
    )


def _shard(tmp_path, weights, **kwargs):
    from soup_cli.utils.layer_shard import shard_checkpoint

    return shard_checkpoint(
        weights, str(tmp_path / "shards"), dtype="float32", arch="deepseek_v3",
        num_hidden_layers=3, **kwargs,
    )


def _paired(tmp_path):
    import torch
    from peft import get_peft_model

    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights, resident = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=str(tmp_path / "shards"), index=index,
        lora_config=_lora(), device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
    )
    ref = get_peft_model(resident, _lora())
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    src = {k.replace(".inner.", "."): v for k, v in model.state_dict().items() if "lora_" in k}
    dst = {k: v for k, v in ref.state_dict().items() if "lora_" in k}
    assert src and set(src) == set(dst)
    for key, value in src.items():
        dst[key].copy_(value)
    model.eval()
    ref.eval()
    ids = torch.randint(0, 128, (2, 12), generator=generator)
    return model, runtime, ref, ids, index


def test_per_expert_layout_moves_routed_experts_out_of_the_layer_file(tmp_path):
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import expert_paths, layer_paths, read_shard_index

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    assert index.expert_layout == "per_expert"
    assert index.expert_layers == (1, 2)
    assert index.expert_count == 8
    assert dict(index.expert_shapes) == {"gate_up": (64, 64), "down": (64, 32)}
    assert read_shard_index(str(tmp_path / "shards")) == index
    layer_one = load_file(layer_paths(str(tmp_path / "shards"), index)[1])
    assert not any(key.startswith("mlp.experts") for key in layer_one)
    assert "mlp.shared_experts.gate_proj.weight" in layer_one
    assert set(expert_paths(str(tmp_path / "shards"), index)) == {1, 2}


def test_streamed_per_expert_forward_matches_resident_eager_bit_exactly(tmp_path):
    import torch

    model, runtime, ref, ids, _ = _paired(tmp_path)
    with torch.no_grad():
        got = model(input_ids=ids, labels=ids)
        want = ref(input_ids=ids, labels=ids)
    assert torch.equal(got.logits, want.logits), (got.logits - want.logits).abs().max().item()
    assert torch.equal(got.loss, want.loss)
    assert runtime.stats()["expert_fetches"] > 0
    runtime.close()


def test_streamed_per_expert_backward_matches_resident_lora_gradients(tmp_path):
    import torch

    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    model, runtime, ref, ids, _ = _paired(tmp_path)
    model(input_ids=ids, labels=ids).loss.backward()
    ref(input_ids=ids, labels=ids).loss.backward()

    def grads(net):
        return {
            name: parameter.grad
            for name, parameter in canonical_named_parameters(net)
            if "lora_" in name and parameter.grad is not None
        }

    streamed, resident = grads(model), grads(ref)
    assert streamed and set(streamed) == set(resident)
    # Layer 0's adapters sit below both MoE layers: their gradient crossed
    # every streamed expert's backward.
    assert any(".layers.0." in name and float(g.abs().sum()) > 0 for name, g in streamed.items())
    for name, gradient in streamed.items():
        assert torch.equal(gradient, resident[name]), name
    runtime.close()


def test_layout_change_is_a_cache_miss_naming_the_layout(tmp_path):
    from soup_cli.utils.layer_shard import inspect_shard_cache

    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    cached, reason = inspect_shard_cache(
        str(tmp_path / "shards"), "float32", index.source_fingerprint, index.source_files,
        "none", False, "", "", stripe_roots=(), expert_layout="",
    )
    assert cached is None
    assert "expert layout changed" in reason
    reused = _shard(tmp_path, weights, expert_layout="per_expert")
    assert reused == index


def test_per_expert_layout_is_refused_for_an_unadmitted_arch(tmp_path):
    from soup_cli.utils.layer_shard import shard_checkpoint

    with pytest.raises(ValueError, match="admitted: deepseek_v3"):
        shard_checkpoint(
            str(tmp_path), str(tmp_path / "s"), dtype="float32", arch="qwen4_exp",
            expert_layout="per_expert",
        )


def test_unknown_expert_layout_is_refused(tmp_path):
    from soup_cli.utils.layer_shard import shard_checkpoint

    with pytest.raises(ValueError, match="unsupported expert_layout"):
        shard_checkpoint(
            str(tmp_path), str(tmp_path / "s"), dtype="float32", arch="deepseek_v3",
            expert_layout="experts",
        )


def test_install_refuses_a_model_whose_experts_disagree_with_the_file(tmp_path):
    from soup_cli.utils.expert_stream import ExpertFetcher, install_expert_streaming

    module, n_experts, _ = _deepseek_v3_experts(5)
    _, reader = _streamed_copy(tmp_path, module, n_experts)
    reader.count = 4  # what a stale index would claim
    holder = _holder(module)
    with pytest.raises(ValueError, match="reshard the checkpoint"):
        install_expert_streaming(
            [holder], {0: reader}, ExpertFetcher(device="cpu", dtype="float32")
        )
    reader.close()


def _holder(experts):
    import torch.nn as nn

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = experts

    return Layer()


# --------------------------------------------------------------------------
# NF4 experts (CUDA)
# --------------------------------------------------------------------------
@CUDA
def test_nf4_streamed_experts_equal_the_dequantised_resident_block(tmp_path):
    import torch
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    from soup_cli.utils.expert_stream import (
        ExpertFetcher,
        ExpertShardReader,
        streamed_experts_class,
        write_expert_shard,
    )
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    module, n_experts, top_k = _deepseek_v3_experts(5)
    module = module.to("cuda", torch.bfloat16)
    path = str(tmp_path / "experts_001.safetensors")
    pairs = [
        (module.gate_up_proj[e].detach(), module.down_proj[e].detach())
        for e in range(n_experts)
    ]
    info = write_expert_shard(path, iter(pairs), quantise=True, double_quant=True, device="cuda")
    with torch.no_grad():
        for e, (gate_up, down) in enumerate(pairs):
            for target, tensor in ((module.gate_up_proj, gate_up), (module.down_proj, down)):
                packed, state = quantize_4bit(
                    tensor, blocksize=64, compress_statistics=True, quant_type="nf4"
                )
                target[e].copy_(dequantize_4bit(packed, quant_state=state))
    codes = {NF4_CODE_KEY: info.codes[0].cuda(), NF4_NESTED_CODE_KEY: info.codes[1].cuda()}
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs=info.quant_specs, dtype="bfloat16"
    )
    streamed = streamed_experts_class()(
        reader, ExpertFetcher(device="cuda", dtype="bfloat16", codes=codes), module.act_fn, 1
    )
    states, index, weights, _ = _routing(24, n_experts, top_k, 32, 9)
    states = states.detach().to("cuda", torch.bfloat16)
    index = index.cuda()
    weights = weights.detach().to("cuda", torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(streamed(states, index, weights), module(states, index, weights))
    reader.close()


# --------------------------------------------------------------------------
# the synthetic store writer and the one-step harness (benchmarks/harness)
# --------------------------------------------------------------------------
def _harness(name):
    import importlib.util
    import sys
    from pathlib import Path

    harness = Path(__file__).resolve().parent.parent / "benchmarks" / "harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    spec = importlib.util.spec_from_file_location(name, harness / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_tiny_store_is_a_cache_the_runtime_accepts(tmp_path):
    """The writer skips the sharder, so its index and files must be exactly
    what the sharder would write: the index reads back, every experts file
    opens with the index's shapes and NF4 specs, and every linear has its
    sidecars."""
    from safetensors import safe_open

    from soup_cli.utils.expert_stream import ExpertShardReader
    from soup_cli.utils.layer_shard import expert_paths, layer_paths, read_shard_index

    store = str(tmp_path / "store")
    summary = _harness("synth_moe_store").write_store(store, "tiny", log=lambda _line: None)
    assert summary["label"] == "random weights, shape only"
    index = read_shard_index(store)
    assert index.arch == "deepseek_v3" and index.quant == "nf4"
    assert index.expert_layers == (1, 2) and index.expert_count == 8
    for layer, path in expert_paths(store, index).items():
        reader = ExpertShardReader(
            path, count=index.expert_count, shapes=index.expert_shapes,
            quant_specs=index.expert_quant_specs, dtype=index.dtype,
        )
        reader.close()
    with safe_open(layer_paths(store, index)[1], framework="pt") as handle:
        keys = set(handle.keys())
    assert "self_attn.q_a_proj.weight::absmax" in keys
    assert "mlp.shared_experts.down_proj.weight" in keys
    assert "mlp.gate.e_score_correction_bias" in keys


def test_synthetic_writer_refuses_an_existing_cache(tmp_path, capsys):
    store = tmp_path / "store"
    store.mkdir()
    (store / "index.json").write_text("{}", encoding="utf-8")
    assert _harness("synth_moe_store").main(["--out", str(store), "--shape", "tiny"]) == 2
    assert "already holds a shard cache" in capsys.readouterr().out


@CUDA
def test_one_step_on_the_synthetic_tiny_store(tmp_path):
    import json

    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    store = str(tmp_path / "store")
    _harness("synth_moe_store").write_store(store, "tiny", log=lambda _line: None)
    out = tmp_path / "step.json"
    assert _harness("moe_stream_step").main(
        ["--store", store, "--out", str(out), "--seq", "16", "--steps", "2", "--no-pin"]
    ) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["label"] == "random weights, shape only"
    assert len(report["steps"]) == 2
    for step in report["steps"]:
        assert step["loss_finite"] and step["grads_finite"]
        assert step["expert_fetches"] > 0
