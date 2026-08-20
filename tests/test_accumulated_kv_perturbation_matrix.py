from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "accumulated_kv_perturbation_matrix.py"
SPEC = importlib.util.spec_from_file_location("accumulated_kv_perturbation_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        capture_output=True,
    )


def plan_args(tmp_path: Path) -> list[str]:
    qxqxf = tmp_path / "qxqxf.exe"
    model = tmp_path / "model.qxf"
    qxqxf.write_bytes(b"binary")
    model.write_bytes(b"model")
    return [
        "plan",
        "--qxqxf",
        str(qxqxf),
        "--model",
        str(model),
        "--experiment-dir",
        str(tmp_path / "experiment"),
        "--revision",
        "0" * 40,
        "--prompt-token",
        "42",
        "--prefix-steps",
        "2",
        "--continuation-steps",
        "1",
        "--kv-formats=f16,int8",
        "--scales=-1,0,1",
        "--activation-format",
        "q8_k_compat",
        "--layers",
        "3",
        "--start-layer",
        "1",
        "--expected-count",
        "4",
        "--direction-index",
        "1",
        "--direction-magnitude",
        "0.125",
        "--ctx",
        "8",
        "--seed",
        "7",
    ]


def test_plan_builds_complete_ordered_matrix_and_records_provenance(tmp_path):
    args = plan_args(tmp_path)

    completed = run_script(*args)

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(completed.stdout)
    assert manifest["schema"] == "qx-accumulated-kv-perturbation-manifest-v1"
    assert manifest["experiment"] == {
        "prompt_token": 42,
        "prefix_steps": 2,
        "continuation_steps": 1,
        "kv_formats": ["f16", "int8"],
        "scales": [-1.0, 0.0, 1.0],
        "activation_format": "q8_k_compat",
        "layers": 3,
        "start_layer": 1,
        "expected_count": 4,
        "direction_index": 1,
        "direction_magnitude": 0.125,
        "runtime_mode": "full_moe",
        "ctx": 8,
        "seed": 7,
    }
    assert [(cell["ordinal"], cell["kv_format"]) for cell in manifest["cells"]] == [
        (0, "f16"),
        (1, "int8"),
    ]
    for cell in manifest["cells"]:
        assert [run["scale"] for run in cell["runs"]] == [-1.0, 0.0, 1.0]
        assert [run["ordinal"] for run in cell["runs"]] == [0, 1, 2]
    assert manifest["artifacts"]["qxqxf"]["sha256"] == sha256_file(tmp_path / "qxqxf.exe")
    assert manifest["artifacts"]["model"]["sha256"] == sha256_file(tmp_path / "model.qxf")
    assert manifest["revision"] == "0" * 40
    assert (tmp_path / "experiment" / "matrix-manifest.json").is_file()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--prefix-steps", "1"),
        ("--continuation-steps", "2"),
        ("--kv-formats", "f16"),
        ("--kv-formats", "f16,f16"),
        ("--kv-formats", "int8,f16"),
        ("--scales", "0,1"),
        ("--scales", "-1,1"),
        ("--scales", "-1,nan,0,1"),
        ("--start-layer", "0"),
        ("--direction-index", "4"),
        ("--direction-magnitude", "nan"),
        ("--direction-magnitude", "0"),
        ("--ctx", "2"),
    ],
)
def test_plan_rejects_contract_violations(tmp_path, flag, value):
    args = plan_args(tmp_path)
    index = next(i for i, item in enumerate(args) if item == flag or item.startswith(f"{flag}="))
    if args[index] == flag:
        args[index + 1] = value
    else:
        args[index] = f"{flag}={value}"

    completed = run_script(*args)

    assert completed.returncode != 0
    assert "Traceback" not in completed.stderr


def test_json_loader_rejects_duplicate_keys_and_non_finite_numbers(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"value":NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        MATRIX.load_json_object(duplicate, "manifest")
    with pytest.raises(ValueError, match="non-finite JSON number"):
        MATRIX.load_json_object(non_finite, "manifest")


def test_runtime_command_combines_accumulated_snapshot_and_residual_replay(tmp_path):
    qxqxf = tmp_path / "qxqxf.exe"
    model = tmp_path / "model.qxf"
    tokens = tmp_path / "tokens.txt"
    for path in (qxqxf, model, tokens):
        path.write_bytes(b"fixture")
    args = MATRIX.argparse.Namespace(
        qxqxf=qxqxf,
        model=model,
        tokens=tokens,
        layers=3,
        ctx=8,
        activation_format="q8_k_compat",
        seed=7,

    )
    command = MATRIX._runtime_command(
        args,
        kv_format="int8",
        prompt_token=17,
        steps=1,
        dump_dir=tmp_path / "dump",
        snapshot_in=tmp_path / "snapshot.bin",
        start_layer=1,
        residual_in=tmp_path / "residual.f32",
    )
    assert command[command.index("--kv") + 1] == "int8"
    assert command[command.index("--kv-snapshot-in") + 1].endswith("snapshot.bin")
    assert command[command.index("--start-layer") + 1] == "1"
    assert command[command.index("--residual-in") + 1].endswith("residual.f32")
    assert command[command.index("--tokens") + 1] == str(tokens)


def valid_runtime_result():
    return {
        "probe": "state_loop",
        "prompt_token": 17,
        "steps": 1,
        "layers_run": 3,
        "position_base": 2,
        "kv_format": "f16",
        "activation_format": "q8_k_compat",
        "tokens": [
            {
                "position": 2,
                "layers": [
                    {"layer": 0, "selected_experts": [0, 1]},
                    {"layer": 1, "selected_experts": [1, 0]},
                    {"layer": 2, "selected_experts": [1, 0]},
                ],
            }
        ],
    }


def validate_runtime(value):
    MATRIX._validate_runtime_result(
        value,
        kv_format="f16",
        activation_format="q8_k_compat",
        prompt_token=17,
        steps=1,
        layers=3,
        position_base=2,
    )


@pytest.mark.parametrize("key", ["prompt_token", "steps", "layers_run", "position_base"])
def test_runtime_validation_rejects_bool_as_integer(key):
    value = valid_runtime_result()
    value[key] = True
    with pytest.raises(ValueError, match=key):
        validate_runtime(value)


def test_runtime_validation_rejects_partial_or_disordered_layers():
    value = valid_runtime_result()
    value["tokens"][0]["layers"] = [
        {"layer": 2, "selected_experts": [1, 0]},
        {"layer": 1, "selected_experts": [1, 0]},
    ]
    with pytest.raises(ValueError, match="partial or disordered"):
        validate_runtime(value)


def test_exact_f32_reader_rejects_nan_inf_and_wrong_size(tmp_path):
    path = tmp_path / "residual.f32"
    path.write_bytes(struct.pack("<2f", 1.0, float("nan")))
    with pytest.raises(ValueError, match="NaN or Inf"):
        MATRIX.read_exact_f32(path, 2)
    path.write_bytes(struct.pack("<f", 1.0))
    with pytest.raises(ValueError, match="expected 8 bytes"):
        MATRIX.read_exact_f32(path, 2)


def test_artifact_validation_rejects_payload_mutation_and_path_mismatch(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"before")
    record = MATRIX._relative_artifact(tmp_path, path)
    path.write_bytes(b"after!")
    with pytest.raises(ValueError, match="hash mismatch"):
        MATRIX._verify_relative_artifact(tmp_path, record, "artifact.bin", "artifact")
    with pytest.raises(ValueError, match="path mismatch"):
        MATRIX._verify_relative_artifact(
            tmp_path,
            {"path": "../escape.bin", "sha256": "0" * 64},
            "artifact.bin",
            "artifact",
        )


def test_analyze_rejects_plan_without_complete_results(tmp_path):
    completed = run_script(*plan_args(tmp_path))
    assert completed.returncode == 0, completed.stderr
    with pytest.raises(ValueError, match="incomplete"):
        MATRIX.analyze_matrix(
            tmp_path / "experiment",
            tmp_path / "qxqxf.exe",
            tmp_path / "model.qxf",
            "0" * 40,
        )


def test_dump_resolver_accepts_absolute_replay_step_and_rejects_ambiguity(tmp_path):
    expected = tmp_path / "step-2-layer-1-output.f32"
    expected.write_bytes(struct.pack("<f", 1.0))
    assert MATRIX._dump_path(tmp_path, 1) == expected
    (tmp_path / "step-3-layer-1-output.f32").write_bytes(struct.pack("<f", 1.0))
    with pytest.raises(ValueError, match="expected one residual dump"):
        MATRIX._dump_path(tmp_path, 1)


def test_scaled_residual_returns_exact_f32_tuple():
    actual = MATRIX.scaled_residual((0.1, -0.25), (0.3, 0.25), 0.5)
    assert isinstance(actual, tuple)
    assert actual == (
        struct.unpack("<f", struct.pack("<f", 0.2))[0],
        0.0,
    )


def test_atomic_write_preserves_existing_destination_when_publish_fails(tmp_path, monkeypatch):
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"trusted-old")

    def fail_replace(source, target):
        raise OSError("injected publish failure")

    monkeypatch.setattr(MATRIX.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected publish failure"):
        MATRIX._atomic_write_bytes(destination, b"partial-new")

    assert destination.read_bytes() == b"trusted-old"
    assert list(tmp_path.glob(".artifact.bin.*.tmp")) == []


def test_run_matrix_happy_path_enforces_zero_control_and_reports_cross_mode(tmp_path, monkeypatch):
    args = MATRIX.parse_args(plan_args(tmp_path))

    def token(position: int, start_layer: int = 0):
        return {
            "position": position,
            "input_token": 17 if position == 2 else 42,
            "layers": [
                {"layer": layer, "selected_experts": [layer, layer + 1]}
                for layer in range(start_layer, 3)
            ],
        }

    def fake_run_json(command, result_path):
        kv_format = command[command.index("--kv") + 1]
        steps = int(command[command.index("--steps") + 1])
        start_layer = int(command[command.index("--start-layer") + 1]) if "--start-layer" in command else 0
        is_snapshot_replay = "--kv-snapshot-in" in command
        position_base = 2 if is_snapshot_replay else 0
        tokens = [token(position_base + index, start_layer) for index in range(steps)]
        result = {
            "probe": "state_loop",
            "prompt_token": int(command[command.index("--prompt-token") + 1]),
            "steps": steps,
            "layers_run": (3 - start_layer) * steps,
            "position_base": position_base,
            "kv_format": kv_format,
            "activation_format": "q8_k_compat",
            "tokens": tokens,
            "final_token": 17 if result_path.parent.name == "capture" else 56,
        }
        if "--kv-snapshot-out" in command:
            Path(command[command.index("--kv-snapshot-out") + 1]).write_bytes(
                f"snapshot-{kv_format}".encode()
            )
        base = (1.0, 2.0, 3.0, 4.0) if kv_format == "f16" else (5.0, 6.0, 7.0, 8.0)
        if result_path.parent.name == "replay":
            MATRIX.write_f32(result_path.parent / "l_out-0.f32", base)
            MATRIX.write_f32(result_path.parent / "l_out-2.f32", base)
        elif "--residual-in" in command:
            residual = MATRIX.read_exact_f32(
                Path(command[command.index("--residual-in") + 1]), 4
            )
            MATRIX.write_f32(result_path.parent / "l_out-2.f32", residual)
        MATRIX.write_json(result_path, result)
        return result

    def fake_snapshot_manifest(run_args, snapshot_path, output_path, prompt_tokens):
        manifest = {
            "producer": {"revision": run_args.revision},
            "geometry": {
                "layers": 3,
                "ctx_tokens": 8,
                "positions": 2,
                "kv_format": "f16" if "kv-f16" in str(snapshot_path) else "int8",
                "activation_format": "q8_k_compat",
                "next_token": 17,
            },
            "payload": {"sha256": sha256_file(snapshot_path)},
        }
        MATRIX.write_json(output_path, manifest)
        return manifest

    monkeypatch.setattr(MATRIX, "_run_json", fake_run_json)
    monkeypatch.setattr(MATRIX, "_build_snapshot_manifest", fake_snapshot_manifest)
    monkeypatch.setattr(MATRIX, "_validate_snapshot_with_helper", lambda *args, **kwargs: None)

    report = MATRIX.run_matrix(args)

    assert report["matrix_complete"] is True
    assert [cell["kv_format"] for cell in report["cells"]] == ["f16", "int8"]
    assert all(cell["control_exact"] is True for cell in report["cells"])
    assert report["experiment"]["runtime_mode"] == "full_moe"
    assert set(report["provenance"]) == {"qxqxf", "model"}
    assert all(set(artifact) == {"sha256"} for artifact in report["provenance"].values())
    assert report["cross_mode"]["formats"] == ["f16", "int8"]
    assert [run["selected_token_parity"] for run in report["cross_mode"]["runs"]] == [
        True,
        True,
        True,
    ]
    assert json.loads((tmp_path / "experiment" / "report.json").read_text()) == report
