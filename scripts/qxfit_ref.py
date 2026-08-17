#!/usr/bin/env python
import argparse, json

SHAPES = {
    "qwen3-4b": dict(layers=36, hidden=2560, q_heads=32, kv_heads=8, head_dim=128, vocab=151936),
    "qwen3-8b": dict(layers=36, hidden=4096, q_heads=32, kv_heads=8, head_dim=128, vocab=151936),
    "qwen3-14b": dict(layers=40, hidden=5120, q_heads=40, kv_heads=8, head_dim=128, vocab=151936),
    "qwen3-30b-a3b": dict(layers=48, hidden=2048, q_heads=32, kv_heads=4, head_dim=128, vocab=151936),
}

def kv_gib(shape, ctx, kv):
    bpv = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}[kv]
    return (2 * shape["layers"] * shape["kv_heads"] * shape["head_dim"] * bpv * ctx) / 1024**3

def plan(args):
    s = SHAPES[args.model]
    usable_ram = max(0.0, args.ram_gib - args.os_reserve_gib)
    usable_vram = max(0.0, args.vram_gib - args.cuda_reserve_gib)
    kv = kv_gib(s, args.ctx, args.kv)
    overhead = kv + args.scratch_gib + args.tokenizer_gib
    total = args.weight_gib + overhead
    vram_after_runtime = max(0.0, usable_vram - overhead)
    vram_weights = min(args.weight_gib, vram_after_runtime)
    ram_weights = args.weight_gib - vram_weights
    feasible = overhead < usable_vram and total <= (usable_ram + usable_vram) and ram_weights <= usable_ram
    return {
        "model": args.model,
        "weight_gib": round(args.weight_gib, 3),
        "ctx_tokens": args.ctx,
        "kv_format": args.kv,
        "kv_gib": round(kv, 3),
        "runtime_overhead_gib": round(overhead, 3),
        "total_active_gib": round(total, 3),
        "usable_ram_gib": round(usable_ram, 3),
        "usable_vram_gib": round(usable_vram, 3),
        "suggested_vram_weights_gib": round(vram_weights, 3),
        "suggested_ram_weights_gib": round(ram_weights, 3),
        "feasible": feasible,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=SHAPES, default="qwen3-8b")
    p.add_argument("--weight-gib", type=float, default=3.3)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--kv", choices=["fp16", "int8", "int4"], default="int8")
    p.add_argument("--ram-gib", type=float, default=14.0)
    p.add_argument("--vram-gib", type=float, default=5.0)
    p.add_argument("--os-reserve-gib", type=float, default=3.0)
    p.add_argument("--cuda-reserve-gib", type=float, default=0.8)
    p.add_argument("--scratch-gib", type=float, default=0.35)
    p.add_argument("--tokenizer-gib", type=float, default=0.12)
    args = p.parse_args()
    print(json.dumps(plan(args), indent=2))

if __name__ == "__main__":
    main()
