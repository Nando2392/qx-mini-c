"""Deterministic end-to-end smoke check for the QXF C runtime."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "build" / "qxqxf.exe"


def run(*args: object) -> str:
    return subprocess.check_output([str(arg) for arg in args], cwd=ROOT, text=True)


def main() -> int:
    if not EXE.exists():
        subprocess.check_call(["cmd.exe", "/c", "build_msvc.bat"], cwd=ROOT)

    with tempfile.TemporaryDirectory(prefix="qx-smoke-") as temp:
        temp_path = Path(temp)
        gguf = temp_path / "synthetic.gguf"
        qxf = temp_path / "synthetic.qxf"
        tokens = temp_path / "tokens.tsv"

        run(sys.executable, ROOT / "scripts" / "make_synthetic_gguf.py", "--out", gguf)
        run(EXE, "create-from-gguf-copy", "--in", gguf, "--model", "qwen3-30b-a3b", "--quant", "q2", "--out", qxf)
        run(EXE, "tokenizer-export", "--gguf", gguf, "--out", tokens)
        result = json.loads(
            run(
                EXE,
                "state-loop-probe",
                "--in",
                qxf,
                "--tokens",
                tokens,
                "--prompt-token",
                42,
                "--steps",
                2,
                "--layers",
                2,
                "--ctx",
                16,
                "--kv",
                "int8",
                "--top-k",
                3,
                "--scan",
                32,
                "--temperature",
                0,
                "--seed",
                7,
                "--rope-gqa-attention",
                "--residual-dims",
                1152,
                "--norm",
                "blk.0.attn_norm.weight",
            )
        )
        golden = json.loads(run(EXE, "rope-gqa-golden-probe", "--tokens", 2, "--q-heads-run", 9, "--seed", 7))

    real_model = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
    real_golden = None
    if real_model.exists():
        real_golden = json.loads(run(EXE, "real-qkv-golden-probe", "--in", real_model, "--layer", 0, "--token-a", 42, "--token-b", 43, "--q-heads-run", 32, "--seed", 7, "--full-moe"))

    layer0 = result["tokens"][0]["layers"][0]
    assert result["delta_source"] == "rope_gqa_attention"
    assert result["cache_readback_ok"] is True
    assert result["layers_run"] == 4
    assert result["kv_appends"] == 4
    assert layer0["persistent_kv"] is True
    assert layer0["kv_scale_source"] == "dynamic_per_vector"
    assert layer0["k_scale"] > 0
    assert layer0["v_scale"] > 0
    assert abs(layer0["softmax_sum"] - 1.0) < 1e-6
    assert layer0["attention_mode"] == "rope_gqa_full_heads"
    assert layer0["rope_applied"] is True
    assert layer0["q_heads_run"] == 2
    assert layer0["kv_heads_touched"] == 1
    assert layer0["gqa_group_size"] == 8
    assert abs(layer0["softmax_sum_min"] - 1.0) < 1e-6
    assert abs(layer0["softmax_sum_max"] - 1.0) < 1e-6
    assert layer0["attention_output_vector_checksum"] > 0
    assert golden["probe"] == "rope_gqa_golden"
    assert golden["rope_layout"] == "qwen_split_half"
    assert golden["q_heads_run"] == 9
    assert golden["kv_heads_touched"] == 2
    assert abs(golden["softmax_sum_min"] - 1.0) < 1e-9
    assert abs(golden["softmax_sum_max"] - 1.0) < 1e-9
    if real_golden is not None:
        assert real_golden["probe"] == "real_qkv_golden"
        assert real_golden["projection_layout"] == "contiguous_tensor_rows"
        assert real_golden["projection_input_dims"] == 2048
        assert real_golden["projection_blocks_per_row"] == 8
        assert real_golden["q_heads_run"] == 32
        assert real_golden["kv_heads_total"] == 4
        assert real_golden["full_head_coverage"] is True
        assert real_golden["post_attention_norm_tensor"] == "blk.0.ffn_norm.weight"
        assert len(real_golden["selected_experts"]) == 8
        assert real_golden["moe_mode"] == "real_top8_swiglu"
        assert real_golden["experts_run"] == 8
        assert real_golden["moe_output_l2"] > 0
        assert len(real_golden["layer_output_raw"]) == 2048

    print(
        json.dumps(
            {
                "status": "pass",
                "probe": result["probe"],
                "delta_source": result["delta_source"],
                "layers_run": result["layers_run"],
                "kv_appends": result["kv_appends"],
                "cache_readback_ok": result["cache_readback_ok"],
                "softmax_sum": layer0["softmax_sum"],
                "golden_probe": golden["probe"],
                "real_qkv_golden": real_golden is not None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
