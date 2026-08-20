import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scaled_residual_matrix.py"
SPEC = importlib.util.spec_from_file_location("scaled_residual_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX_SCRIPT)
TOKENS = [42, 9707]
ACTIVATIONS = ["f32", "q8_k_compat"]
KV_FORMATS = ["f16", "int8"]
SCALES = [-1.0, 0.0, 1.0]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        capture_output=True,
    )


def run_analyze(
    experiment: Path, manifest: dict[str, object]
) -> subprocess.CompletedProcess[str]:
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    arguments = ["analyze", "--experiment-dir", str(experiment)]
    for name, flag in (
        ("qxqxf", "--qxqxf"),
        ("model", "--model"),
        ("llama_oracle", "--llama-oracle"),
        ("gguf", "--gguf"),
    ):
        entry = artifacts[name]
        assert isinstance(entry, dict)
        arguments.extend((flag, str(entry["path"])))
    return run_script(*arguments)


def cell_slug(token: int, activation: str, kv_format: str) -> str:
    return f"token-{token}/activation-{activation}/kv-{kv_format}"


def cell_report(token: int, activation: str, kv_format: str) -> dict[str, object]:
    membership_transition = token == 9707 and activation == "f32" and kv_format == "int8"
    order_transition = (
        token == 42 and activation == "q8_k_compat" and kv_format == "f16"
    ) or membership_transition
    return {
        "schema": "qx-scaled-residual-replay-v1",
        "matrix": {
            "layers": 3,
            "start_layer": 1,
            "expected_count": 2,
            "kv_format": kv_format,
            "activation_format": activation,
            "prompt_token": token,
            "ctx": 4,
            "temperature": 0.0,
            "seed": 7,
            "residual_formula": "oracle_layer_input + scale * (qx_baseline_layer_input - oracle_layer_input)",
        },
        "direction": {"max_abs": 0.25, "rmse": 0.2, "l2": 0.3, "cosine": 0.99},
        "rows": [
            {
                "scale": scale,
                "directory": f"scale-{scale:g}",
                "effective_input_delta": {
                    "max_abs": abs(scale),
                    "rmse": abs(scale),
                    "l2": abs(scale),
                    "cosine": 1.0,
                },
                "final_vs_oracle": {
                    "max_abs": 0.1,
                    "rmse": 0.1,
                    "l2": 0.1,
                    "cosine": 1.0,
                },
                "final_vs_scale_zero": {
                    "max_abs": abs(scale),
                    "rmse": abs(scale),
                    "l2": abs(scale),
                    "cosine": 1.0,
                },
                "requested_direction_fit": {
                    "projected_scale": scale,
                    "cosine": 1.0,
                    "error_l2": 0.0,
                },
                "suffix_response_gain_l2": None if scale == 0.0 else 1.0,
                "routing_order_transition_count": int(order_transition and scale == 1.0),
                "routing_order_transition_layers": (
                    [2] if order_transition and scale == 1.0 else []
                ),
                "routing_membership_transition_count": int(
                    membership_transition and scale == 1.0
                ),
                "routing_membership_transition_layers": (
                    [2] if membership_transition and scale == 1.0 else []
                ),
            }
            for scale in SCALES
        ],
        "routing": {
            "reference_scale": 0.0,
            "order_transition_scales": [1.0] if order_transition else [],
            "membership_transition_scales": [1.0] if membership_transition else [],
        },
        "verdict": (
            "topk_membership_transition_observed"
            if membership_transition
            else "topk_rank_order_transition_observed"
            if order_transition
            else "no_topk_transition_observed"
        ),
        "limitation": "One token at position zero.",
    }


def write_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    artifacts = {}
    for name in ("qxqxf", "model", "llama_oracle", "gguf"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        artifacts[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}

    cells = []
    for token in TOKENS:
        for activation in ACTIVATIONS:
            for kv_format in KV_FORMATS:
                slug = cell_slug(token, activation, kv_format)
                report_path = experiment / "cells" / slug / "report.json"
                report_path.parent.mkdir(parents=True)
                report_path.write_text(
                    json.dumps(cell_report(token, activation, kv_format)),
                    encoding="utf-8",
                )
                cells.append(
                    {
                        "prompt_token": token,
                        "activation_format": activation,
                        "kv_format": kv_format,
                        "oracle_kv_format": "q8_0" if kv_format == "int8" else kv_format,
                        "directory": f"cells/{slug}",
                        "report": {
                            "path": str(report_path.resolve()),
                            "sha256": sha256_file(report_path),
                        },
                    }
                )
    manifest = {
        "schema": "qx-scaled-residual-matrix-manifest-v1",
        "matrix": {
            "prompt_tokens": list(TOKENS),
            "activation_formats": list(ACTIVATIONS),
            "kv_formats": list(KV_FORMATS),
            "scales": list(SCALES),
            "layers": 3,
            "start_layer": 1,
            "expected_count": 2,
            "ctx": 4,
            "seed": 7,
        },
        "artifacts": artifacts,
        "cells": cells,
    }
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return experiment, manifest


def test_plan_builds_exact_cross_product_and_records_provenance(tmp_path):
    paths = {}
    for name in ("qxqxf", "model", "llama_oracle", "gguf"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
    experiment = tmp_path / "experiment"

    completed = run_script(
        "plan",
        "--qxqxf",
        str(paths["qxqxf"]),
        "--model",
        str(paths["model"]),
        "--llama-oracle",
        str(paths["llama_oracle"]),
        "--gguf",
        str(paths["gguf"]),
        "--experiment-dir",
        str(experiment),
        "--tokens=42,9707,0",
        "--activations=f32,q8_k_compat",
        "--kv-formats=f16,f32,int8",
        "--scales=0,1",
        "--layers",
        "3",
        "--start-layer",
        "1",
        "--expected-count",
        "2",
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(completed.stdout)
    assert manifest["schema"] == "qx-scaled-residual-matrix-manifest-v1"
    assert len(manifest["cells"]) == 18
    assert len({cell["directory"] for cell in manifest["cells"]}) == 18
    int8_cells = [cell for cell in manifest["cells"] if cell["kv_format"] == "int8"]
    assert {cell["oracle_kv_format"] for cell in int8_cells} == {"q8_0"}
    assert manifest["artifacts"]["model"]["sha256"] == sha256_file(paths["model"])
    assert (experiment / "matrix-manifest.json").is_file()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--tokens", "42,42"),
        ("--activations", "f32,f32"),
        ("--kv-formats", "f16,f16"),
        ("--tokens", "42,-1"),
        ("--activations", "f32,unknown"),
        ("--kv-formats", "f16,int4"),
    ],
)
def test_plan_rejects_invalid_or_duplicate_dimensions(tmp_path, flag, value):
    artifacts = []
    for name in ("qxqxf", "model", "llama", "gguf"):
        path = tmp_path / name
        path.write_bytes(b"x")
        artifacts.append(path)
    arguments = [
        "plan",
        "--qxqxf",
        str(artifacts[0]),
        "--model",
        str(artifacts[1]),
        "--llama-oracle",
        str(artifacts[2]),
        "--gguf",
        str(artifacts[3]),
        "--experiment-dir",
        str(tmp_path / "experiment"),
        "--tokens=42,9707",
        "--activations=f32,q8_k_compat",
        "--kv-formats=f16,int8",
        "--scales=0,1",
        "--layers",
        "3",
        "--start-layer",
        "1",
        "--expected-count",
        "2",
    ]
    option_index = next(i for i, item in enumerate(arguments) if item.startswith(flag))
    if arguments[option_index] == flag:
        arguments[option_index + 1] = value
    else:
        arguments[option_index] = f"{flag}={value}"

    completed = run_script(*arguments)

    assert completed.returncode != 0
    assert "Traceback" not in completed.stderr


def test_analyze_aggregates_complete_matrix_and_keeps_transition_kinds_separate(tmp_path):
    experiment, manifest = write_fixture(tmp_path)

    completed = run_analyze(experiment, manifest)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["schema"] == "qx-scaled-residual-matrix-v1"
    assert report["summary"]["cell_count"] == 8
    assert report["summary"]["order_transition_cell_count"] == 2
    assert report["summary"]["membership_transition_cell_count"] == 1
    cells = {
        (cell["prompt_token"], cell["activation_format"], cell["kv_format"]): cell
        for cell in report["cells"]
    }
    assert cells[(42, "q8_k_compat", "f16")]["order_transition_scales"] == [1.0]
    assert cells[(42, "q8_k_compat", "f16")]["membership_transition_scales"] == []
    assert cells[(9707, "f32", "int8")]["membership_transition_scales"] == [1.0]
    assert len(cells[(42, "q8_k_compat", "f16")]["response_curve"]) == 3
    assert cells[(42, "q8_k_compat", "f16")]["unit_response"] == {
        "minus_one_l2": 1.0,
        "plus_one_l2": 1.0,
        "max_to_min_l2_ratio": 1.0,
    }
    assert "independent one-token" in report["limitation"]


def test_analyze_rejects_manifest_artifact_path_before_reading_it(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    trusted_manifest = json.loads(json.dumps(manifest))
    arbitrary_file = tmp_path / "arbitrary-local-file"
    arbitrary_file.write_bytes(b"must not be selected by the manifest")
    artifact = manifest["artifacts"]["qxqxf"]
    artifact["path"] = str(arbitrary_file.resolve())
    artifact["sha256"] = sha256_file(arbitrary_file)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, trusted_manifest)

    assert completed.returncode != 0
    assert "matrix artifact path mismatch: qxqxf" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize("field", ["expected_count", "prompt_tokens", "scales"])
def test_analyze_rejects_boolean_manifest_numeric_values(tmp_path, field):
    experiment, manifest = write_fixture(tmp_path)
    trusted_manifest = json.loads(json.dumps(manifest))
    if field == "expected_count":
        manifest["matrix"][field] = True
    else:
        manifest["matrix"][field][0] = True
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, trusted_manifest)

    assert completed.returncode != 0
    assert "invalid matrix dimensions" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize("invalid_token", [True, 1.0])
def test_validate_manifest_rejects_non_integer_cell_token(tmp_path, invalid_token):
    experiment = tmp_path / "experiment"
    artifacts = {}
    artifact_entries = {}
    for name in ("qxqxf", "model", "llama_oracle", "gguf"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        artifacts[name] = path
        artifact_entries[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    cell_dir = experiment / "cells" / f"token-{invalid_token}" / "activation-f32" / "kv-f16"
    cell_dir.mkdir(parents=True)
    report = cell_dir / "cell-report.json"
    report.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema": "qx-scaled-residual-matrix-manifest-v1",
        "matrix": {
            "prompt_tokens": [1],
            "activation_formats": ["f32"],
            "kv_formats": ["f16"],
            "scales": [0.0],
            "layers": 2,
            "start_layer": 1,
            "expected_count": 1,
            "ctx": 1,
            "seed": 1,
        },
        "artifacts": artifact_entries,
        "cells": [
            {
                "prompt_token": invalid_token,
                "activation_format": "f32",
                "kv_format": "f16",
                "oracle_kv_format": "f16",
                "directory": f"cells/token-{invalid_token}/activation-f32/kv-f16",
                "report": {"path": str(report.resolve()), "sha256": sha256_file(report)},
            }
        ],
    }

    with pytest.raises(ValueError, match="invalid matrix cell metadata"):
        MATRIX_SCRIPT.validate_manifest(manifest, experiment, artifacts)


@pytest.mark.parametrize(
    "field",
    ["schema", "token_id", "n_embd", "n_layer", "layer", "layer_count", "final_count"],
)
@pytest.mark.parametrize("invalid_value", [True, 1.0])
def test_validate_oracle_result_rejects_non_integer_metadata(field, invalid_value):
    result = {
        "schema": 1,
        "ok": True,
        "token_id": 1,
        "kv_type": "f16",
        "n_embd": 1,
        "n_layer": 1,
        "layers": [{"layer": 0, "count": 1, "written": True}],
        "internals": [{"name": "l_out-0", "count": 1, "written": True}],
    }
    if field in {"schema", "token_id", "n_embd", "n_layer"}:
        result[field] = invalid_value
    elif field == "layer":
        result["layers"][0]["layer"] = invalid_value
    elif field == "layer_count":
        result["layers"][0]["count"] = invalid_value
    else:
        result["internals"][0]["count"] = invalid_value

    with pytest.raises(ValueError):
        MATRIX_SCRIPT.validate_oracle_result(
            result,
            token=1,
            oracle_kv="f16",
            matrix={"start_layer": 0, "layers": 1, "expected_count": 1},
        )


@pytest.mark.parametrize(
    "field",
    [
        "prompt_token",
        "prompt_token_count",
        "prompt_token_ids",
        "generation_steps",
        "steps",
        "layers",
        "start_layer",
        "ctx_tokens",
        "layers_run",
        "kv_appends",
    ],
)
@pytest.mark.parametrize("invalid_value", [True, 1.0])
def test_validate_baseline_result_rejects_non_integer_metadata(field, invalid_value):
    result = {
        "probe": "state_loop",
        "prompt_token": 1,
        "prompt_token_count": 1,
        "prompt_token_ids": [1],
        "generation_steps": 1,
        "steps": 1,
        "layers": 2,
        "start_layer": 0,
        "ctx_tokens": 4,
        "kv_format": "f16",
        "activation_format": "f32",
        "residual_dump": True,
        "layers_run": 2,
        "kv_appends": 2,
        "cache_readback_ok": True,
    }
    result[field] = [invalid_value] if field == "prompt_token_ids" else invalid_value

    with pytest.raises(ValueError, match="baseline result metadata mismatch"):
        MATRIX_SCRIPT.validate_baseline_result(
            result,
            token=1,
            activation="f32",
            kv_format="f16",
            matrix={"layers": 2, "ctx": 4},
        )


@pytest.mark.parametrize(
    "field", ["prompt_token", "layers", "start_layer", "expected_count", "ctx", "seed"]
)
@pytest.mark.parametrize("invalid_value", [True, 1.0])
def test_analyze_rejects_non_integer_cell_report_metadata(
    tmp_path, field, invalid_value
):
    experiment, manifest = write_fixture(tmp_path)
    cell = manifest["cells"][0]
    report_path = Path(cell["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["matrix"][field] = invalid_value
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "cell metadata mismatch" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_analyze_rejects_boolean_cell_scale(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    cell = manifest["cells"][0]
    report_path = Path(cell["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"][-1]["scale"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "invalid scale row" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_analyze_rejects_boolean_routing_transition_count(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    cell = manifest["cells"][0]
    report_path = Path(cell["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"][0]["routing_order_transition_count"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "routing metrics are invalid" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_analyze_rejects_boolean_routing_transition_scale(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    for cell in manifest["cells"]:
        report_path = Path(cell["report"]["path"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["routing"]["order_transition_scales"]:
            break
    else:
        raise AssertionError("fixture must contain a routing-order transition")
    report["routing"]["order_transition_scales"][0] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "routing transition scales are invalid" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_analyze_rejects_boolean_routing_reference_scale(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    cell = manifest["cells"][0]
    report_path = Path(cell["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["routing"]["reference_scale"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "routing summary is invalid" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_analyze_rejects_partial_matrix(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    manifest["cells"].pop()
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "complete matrix" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_analyze_rejects_non_object_cell_report(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    cell = manifest["cells"][0]
    report_path = Path(cell["report"]["path"])
    report_path.write_text("[]", encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "report root must be a JSON object" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_analyze_rejects_non_finite_cell_metric(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    cell = manifest["cells"][0]
    report_path = Path(cell["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["direction"]["l2"] = math.nan
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "non-finite" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_analyze_rejects_cell_metadata_mismatch(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    cell = manifest["cells"][0]
    report_path = Path(cell["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["matrix"]["prompt_token"] = 123
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "cell metadata mismatch" in completed.stderr


def test_analyze_rejects_cell_report_missing_direction(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    cell = manifest["cells"][0]
    report_path = Path(cell["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("direction")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "direction metrics" in completed.stderr


def test_analyze_rejects_cell_report_with_partial_rows(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    cell = manifest["cells"][0]
    report_path = Path(cell["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"][0].pop("final_vs_oracle")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "row metrics" in completed.stderr


def test_analyze_rejects_report_hash_mismatch(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    report_path = Path(manifest["cells"][0]["report"]["path"])
    report_path.write_text(report_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "cell report hash mismatch" in completed.stderr


def test_analyze_rejects_artifact_hash_mismatch(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    Path(manifest["artifacts"]["model"]["path"]).write_bytes(b"changed")

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "matrix artifact hash mismatch: model" in completed.stderr


def test_analyze_rejects_incomplete_scale_rows(tmp_path):
    experiment, manifest = write_fixture(tmp_path)
    cell = manifest["cells"][0]
    report_path = Path(cell["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"].pop()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    cell["report"]["sha256"] = sha256_file(report_path)
    (experiment / "matrix-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    completed = run_analyze(experiment, manifest)

    assert completed.returncode != 0
    assert "complete scale matrix" in completed.stderr
    assert "Traceback" not in completed.stderr
