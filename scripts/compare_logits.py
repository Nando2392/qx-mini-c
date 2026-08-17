#!/usr/bin/env python3
"""Compare complete QX and llama.cpp F32 logit sidecars."""

import argparse
import json
import math
import sys
from pathlib import Path

from compare_residuals import compare_values, read_f32


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qx", required=True, type=Path)
    parser.add_argument("--llama", required=True, type=Path)
    parser.add_argument("--max-abs", type=float, default=math.inf)
    parser.add_argument("--rmse", type=float, default=math.inf)
    parser.add_argument("--min-cosine", type=float, default=-1.0)
    args = parser.parse_args()

    try:
        qx_values = read_f32(args.qx)
        llama_values = read_f32(args.llama)
        result = compare_values(qx_values, llama_values)
        qx_argmax = argmax(qx_values)
        llama_argmax = argmax(llama_values)
        passed = (
            result["max_abs"] <= args.max_abs
            and result["rmse"] <= args.rmse
            and result["cosine"] >= args.min_cosine
            and qx_argmax == llama_argmax
        )
        payload = {
            "qx": str(args.qx),
            "llama": str(args.llama),
            "count": len(qx_values),
            **result,
            "qx_argmax": qx_argmax,
            "llama_argmax": llama_argmax,
            "argmax_match": qx_argmax == llama_argmax,
            "thresholds": {
                "max_abs": args.max_abs,
                "rmse": args.rmse,
                "min_cosine": args.min_cosine,
            },
            "pass": passed,
        }
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
