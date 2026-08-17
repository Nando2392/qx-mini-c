#!/usr/bin/env python
"""Compara secuencias greedy QX F32/Q8_K para [42] y Hello! contra goldens llama.cpp."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path


LLAMA_GOLDENS = {
    "token-42": {"f16": [1124, 50853], "q8_0": [1124, 50853]},
    "hello": {"f16": [358, 1184], "q8_0": [358, 614]},
}


def invoke(command: list[str]) -> tuple[dict, float]:
    started = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {result.stderr[-2000:]}")
    return json.loads(result.stdout), elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qx-exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.qx_exe, args.model, args.tokenizer):
        if not path.exists():
            parser.error(f"missing fixture: {path}")
    runs = []
    with tempfile.TemporaryDirectory(prefix="qx-q8k-greedy-") as temp:
        prompt = Path(temp) / "hello.txt"
        prompt.write_text("Hello!", encoding="utf-8")
        for activation in ("f32", "q8_k_compat"):
            for kv in ("f32", "int8"):
                common = ["--layers", "48", "--ctx", "4", "--kv", kv, "--activation", activation,
                          "--temperature", "0", "--seed", "7", "--full-moe", "--final-head", "--top-n", "5"]
                token_payload, token_elapsed = invoke([
                    str(args.qx_exe), "state-loop-probe", "--in", str(args.model),
                    "--prompt-token", "42", "--steps", "2", *common,
                ])
                hello_payload, hello_elapsed = invoke([
                    str(args.qx_exe), "prompt-state-loop-probe", "--in", str(args.model),
                    "--tokenizer", str(args.tokenizer), "--text-file", str(prompt), "--generate", "2", *common,
                ])
                token_generated = [step["selected_token"] for step in token_payload["tokens"]]
                hello_generated = [step["selected_token"] for step in hello_payload["tokens"] if step["phase"] == "generate"]
                runs.append({
                    "activation_format": activation,
                    "kv_format": kv,
                    "token_42": token_generated,
                    "hello_prompt_token_ids": hello_payload["prompt_token_ids"],
                    "hello": hello_generated,
                    "token_42_elapsed_seconds": token_elapsed,
                    "hello_elapsed_seconds": hello_elapsed,
                    "matches_llama_f16": {
                        "token_42": token_generated == LLAMA_GOLDENS["token-42"]["f16"],
                        "hello": hello_generated == LLAMA_GOLDENS["hello"]["f16"],
                    },
                })
    print(json.dumps({"schema": 1, "oracle_commit": "768d2a481a99cb75ec9a03b95dadbd35e7acf496", "llama_goldens": LLAMA_GOLDENS, "runs": runs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
