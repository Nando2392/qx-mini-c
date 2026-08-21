from __future__ import annotations

import hashlib
import importlib.util
import math
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "q8k_perf_experiment.py"
SPEC = importlib.util.spec_from_file_location("q8k_perf_experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PERF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PERF)


def native_payload(
    *, activation: str, selected: tuple[int, ...], checksum_bias: int = 0,
    io_backend: str = "buffered",
) -> dict:
    tokens = [
        {"phase": "prefill", "input_token": 9707, "selected_token": None},
        {
            "phase": "generate",
            "input_token": 0,
            "selected_token": selected[0],
            "final_head": {
                "final_residual_checksum": 100 + checksum_bias,
                "final_norm_checksum": 200 + checksum_bias,
                "logits_checksum": 300 + checksum_bias,
                "argmax_token": selected[0],
            },
        },
        {
            "phase": "generate",
            "input_token": selected[0],
            "selected_token": selected[1],
            "final_head": {
                "final_residual_checksum": 101 + checksum_bias,
                "final_norm_checksum": 201 + checksum_bias,
                "logits_checksum": 301 + checksum_bias,
                "argmax_token": selected[1],
            },
        },
    ]
    return {
        "probe": "state_loop",
        "prompt_token_ids": [9707, 0],
        "prompt_token_count": 2,
        "generation_steps": 2,
        "ctx_tokens": 16,
        "kv_format": "int8",
        "activation_format": activation,
        "io_backend": io_backend,
        "layers": 48,
        "cache_readback_ok": True,
        "tokens": tokens,
        "final_token": selected[-1],
        "k_cache_checksum": 400 + checksum_bias,
        "v_cache_checksum": 500 + checksum_bias,
        "bench": {
            "enabled": True,
            "elapsed_sec": 3.0,
            "tokens_per_second": 1.0,
            "ms_per_token": 1000.0,
            "layer_steps": 144,
            "layer_steps_per_second": 48.0,
            "phases": {
                "prefill": {"tokens": 1, "elapsed_sec": 1.0, "tokens_per_second": 1.0},
                "decode": {"tokens": 2, "elapsed_sec": 2.0, "tokens_per_second": 1.0},
            },
        },
    }


def test_summarize_is_fail_closed_and_reports_dispersion():
    summary = PERF.summarize([1.0, 2.0, 3.0])
    assert summary == {
        "count": 3,
        "min": 1.0,
        "median": 2.0,
        "max": 3.0,
        "mad": 1.0,
        "stdev": 1.0,
        "pstdev": pytest.approx(math.sqrt(2.0 / 3.0)),
    }
    for invalid in ([], [0.0, 1.0, 2.0], [1.0, float("nan"), 2.0]):
        with pytest.raises(ValueError):
            PERF.summarize(invalid)


def test_source_state_records_dirty_diff_digest(monkeypatch):
    revision = "a" * 40
    diff = b"binary diff\x00payload"
    results = iter([
        SimpleNamespace(stdout=revision + "\n"),
        SimpleNamespace(stdout=diff),
    ])
    monkeypatch.setattr(PERF.subprocess, "run", lambda *args, **kwargs: next(results))

    assert PERF.source_state() == {
        "revision": revision,
        "working_tree_dirty": True,
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def test_extract_output_signature_requires_complete_final_head_evidence():
    payload = native_payload(activation="f32", selected=(358, 1184))
    signature = PERF.extract_output_signature(payload)
    assert signature["prompt_token_ids"] == [9707, 0]
    assert signature["selected_tokens"] == [358, 1184]
    assert signature["logits_checksums"] == [300, 301]
    assert signature["cache_readback_ok"] is True

    del payload["tokens"][1]["final_head"]["logits_checksum"]
    with pytest.raises(ValueError, match="logits_checksum"):
        PERF.extract_output_signature(payload)


def test_validate_native_payload_rejects_argument_or_phase_drift():
    payload = native_payload(activation="q8_k_compat", selected=(358, 1184))
    signature = PERF.validate_native_payload(
        payload,
        activation="q8_k_compat",
        io_backend="buffered",
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
    )
    assert signature["selected_tokens"] == [358, 1184]

    payload["ctx_tokens"] = 17
    with pytest.raises(ValueError, match="ctx_tokens"):
        PERF.validate_native_payload(
            payload,
            activation="q8_k_compat",
            io_backend="buffered",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )

    wrong_prefill = native_payload(activation="q8_k_compat", selected=(358, 1184))
    wrong_prefill["bench"]["phases"]["prefill"]["tokens"] = 2
    with pytest.raises(ValueError, match="prefill.tokens"):
        PERF.validate_native_payload(
            wrong_prefill,
            activation="q8_k_compat",
            io_backend="buffered",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )

    wrong_backend = native_payload(activation="q8_k_compat", selected=(358, 1184))
    wrong_backend["io_backend"] = "mmap"
    with pytest.raises(ValueError, match="io_backend"):
        PERF.validate_native_payload(
            wrong_backend,
            activation="q8_k_compat",
            io_backend="buffered",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )

    missing_selected = native_payload(activation="q8_k_compat", selected=(358, 1184))
    missing_selected["tokens"][2]["selected_token"] = None
    with pytest.raises(ValueError, match="selected-token count"):
        PERF.validate_native_payload(
            missing_selected,
            activation="q8_k_compat",
            io_backend="buffered",
            kv="int8",
            layers=48,
            ctx=16,
            generation_steps=2,
        )


def test_output_gate_requires_buffered_mmap_equivalence_within_each_activation():
    f32 = PERF.extract_output_signature(native_payload(activation="f32", selected=(358, 1184)))
    f32_mmap = PERF.extract_output_signature(
        native_payload(activation="f32", selected=(358, 1184), io_backend="mmap")
    )
    q8k = PERF.extract_output_signature(
        native_payload(activation="q8_k_compat", selected=(358, 1184), checksum_bias=10)
    )
    q8k_mmap = PERF.extract_output_signature(
        native_payload(activation="q8_k_compat", selected=(358, 1184), checksum_bias=10, io_backend="mmap")
    )
    cells = {
        "buffered:f32": [f32, f32], "mmap:f32": [f32_mmap, f32_mmap],
        "buffered:q8_k_compat": [q8k, q8k], "mmap:q8_k_compat": [q8k_mmap, q8k_mmap],
    }
    gate = PERF.validate_output_contract(cells)
    assert gate["status"] == "pass"
    assert gate["io_equivalence"] == {"f32": True, "q8_k_compat": True}
    assert gate["activation_numeric_checksums_equal"] is False

    changed = PERF.extract_output_signature(
        native_payload(activation="q8_k_compat", selected=(358, 999), checksum_bias=20)
    )
    with pytest.raises(ValueError, match="not deterministic"):
        PERF.validate_output_contract({**cells, "buffered:q8_k_compat": [q8k, changed]})
    with pytest.raises(ValueError, match="buffered/mmap output mismatch"):
        PERF.validate_output_contract({**cells, "mmap:f32": [changed]})


def test_build_inference_command_fixes_real_prompt_and_modal_arguments(tmp_path):
    command = PERF.build_inference_command(
        qx_exe=tmp_path / "qxqxf.exe",
        model=tmp_path / "model.qxf",
        tokenizer=tmp_path / "model.qxt",
        prompt_file=tmp_path / "prompt.txt",
        activation="f32",
        kv="int8",
        layers=48,
        ctx=16,
        generation_steps=2,
        seed=7,
        io_backend="mmap",
    )
    assert command[1] == "prompt-state-loop-probe"
    assert "--full-moe" in command
    assert "--final-head" in command
    assert "--bench" in command
    assert command[command.index("--temperature") + 1] == "0"
    assert command[command.index("--io-backend") + 1] == "mmap"


def test_build_artifact_provenance_pins_source_model_and_runtime_inputs(tmp_path):
    paths = {}
    for name in ("benchmark_script", "qx_exe", "source_model_gguf", "model_qxf", "tokenizer_qxt", "prompt"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths[name] = path

    artifacts = PERF.build_artifact_provenance(**paths)

    assert list(artifacts) == ["benchmark_script", "qx_exe", "source_model_gguf", "model_qxf", "tokenizer_qxt", "prompt"]
    for name, path in paths.items():
        assert artifacts[name]["path"] == str(path.resolve())
        assert artifacts[name]["size"] == len(name)
        assert len(artifacts[name]["sha256"]) == 64
