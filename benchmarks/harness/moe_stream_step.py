"""One measured LoRA training step on an expert-streamed DeepSeek-V3-family store.

Loads a shard cache built with ``expert_layout="per_expert"`` (by the sharder
or by ``synth_moe_store.py``), installs layer + expert streaming, and runs
``--steps`` optimizer steps of LoRA on the attention projections: forward,
backward, AdamW. Writes step seconds, tokens per second, peak VRAM, the
process's peak host memory, and the bytes the expert readers read. On a
synthetic store every number is "random weights, shape only".

    python benchmarks/harness/moe_stream_step.py --store D:/soup-bench/moe-k2-synth \
        --seq 512 --steps 1 --out step.json

With ``--weights`` a real checkpoint is sharded into ``--store`` first (NF4 on
CUDA, the way ``soup train`` does it: native transformers class, the config's
layer count, ``--expert-layout``), unless ``--store`` already holds an index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ATTENTION = ["q_proj", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", required=True, help="shard cache dir")
    parser.add_argument("--weights", default=None, help="checkpoint to shard into --store")
    parser.add_argument(
        "--model", default=None, help="dir with config.json (default: --weights, else --store)"
    )
    parser.add_argument("--expert-layout", choices=("fused", "per_expert"), default="per_expert")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--tier", choices=("ram", "disk"), default="ram")
    parser.add_argument("--no-pin", action="store_true")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--label", default="random weights, shape only")
    return parser.parse_args(argv)


def _shard(args):
    """Shard ``--weights`` into ``--store`` the way the streamed trainer does."""
    from transformers import AutoConfig

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes

    config = AutoConfig.from_pretrained(args.weights, trust_remote_code=False)
    probe = build_meta_skeleton(args.weights, dtype="bfloat16", quant="nf4")
    suffixes = quantised_layer_suffixes(probe)
    del probe
    started = time.perf_counter()
    shard_checkpoint(
        args.weights, args.store, dtype="bfloat16", arch="deepseek_v3", quant="nf4",
        quant_suffixes=suffixes, double_quant=True, quant_device="cuda",
        num_hidden_layers=int(config.num_hidden_layers),
        expert_layout="" if args.expert_layout == "fused" else "per_expert",
        notify=print,
    )
    return round(time.perf_counter() - started, 1)


def _peak_host_bytes():
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", 0) or getattr(info, "rss", 0))
    except ImportError:
        return None


def main(argv=None):
    import torch
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import read_shard_index, shard_cache_disk_bytes
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    args = _args(argv)
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    model_dir = args.model or args.weights or args.store
    report = {"label": args.label, "args": vars(args), "steps": []}
    if args.weights and not os.path.exists(os.path.join(args.store, "index.json")):
        report["shard_seconds"] = _shard(args)
    index = read_shard_index(args.store)
    report["store_bytes"] = shard_cache_disk_bytes(args.store, index)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
        target_modules=_ATTENTION, task_type=TaskType.CAUSAL_LM,
    )
    started = time.perf_counter()
    model, runtime = build_streamed_model(
        model_id=model_dir, shard_dir=args.store, index=index, lora_config=lora,
        device="cuda", dtype=index.dtype, buffers=args.buffers, pin=not args.no_pin,
        tier=args.tier, seed=args.seed, quant=index.quant,
        double_quant=bool(index.double_quant),
    )
    report["build_seconds"] = round(time.perf_counter() - started, 2)
    model.train()
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    report["trainable_params"] = sum(param.numel() for param in trainable)
    vocab = model.config.vocab_size
    generator = torch.Generator().manual_seed(17)
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(args.steps):
            ids = torch.randint(0, vocab, (args.batch, args.seq), generator=generator).cuda()
            before = runtime.stats()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            after = runtime.stats()
            grads_finite = all(
                param.grad is None or bool(torch.isfinite(param.grad).all()) for param in trainable
            )
            report["steps"].append(
                {
                    "step": step,
                    "seconds": round(seconds, 2),
                    "tokens_per_second": round(args.batch * args.seq / seconds, 3),
                    "loss": float(loss.detach()),
                    "loss_finite": bool(torch.isfinite(loss.detach())),
                    "grads_finite": grads_finite,
                    "expert_fetches": after["expert_fetches"] - before["expert_fetches"],
                    "expert_bytes_read": after["expert_bytes_read"] - before["expert_bytes_read"],
                    "layer_loads": after["layer_loads"] - before["layer_loads"],
                    "peak_vram_mib": torch.cuda.max_memory_allocated() // 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20,
                    "peak_host_bytes": _peak_host_bytes(),
                }
            )
            print(json.dumps(report["steps"][-1]))
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
    finally:
        runtime.close()
    report["stats"] = runtime.stats()
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
