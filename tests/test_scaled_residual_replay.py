import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scaled_residual_replay.py"
BASE_EXPERTS = [10, 11, 12, 13, 14, 15, 16, 17]


def write_f32(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        capture_output=True,
    )


def prepare_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    oracle = tmp_path / "oracle"
    baseline = tmp_path / "baseline"
    experiment = tmp_path / "experiment"
    write_f32(oracle / "layer-1.f32", [1.0, 2.0])
    write_f32(oracle / "l_out-2.f32", [10.0, 20.0])
    write_f32(baseline / "step-0-layer-0-output.f32", [1.25, 1.5])
    completed = run_script(
        "prepare",
        "--oracle-dir",
        str(oracle),
        "--baseline-dir",
        str(baseline),
        "--experiment-dir",
        str(experiment),
        "--scales=0,1,2",
        "--layers",
        "3",
        "--start-layer",
        "1",
        "--expected-count",
        "2",
        "--kv-format",
        "f16",
        "--activation-format",
        "q8_k_compat",
    )
    assert completed.returncode == 0, completed.stderr
    return oracle, baseline, experiment


def write_run(run_dir: Path, *, final: list[float], layer2_experts: list[int]) -> None:
    layers = []
    for layer, experts in ((1, [1, 2, 3, 4, 5, 6, 7, 8]), (2, layer2_experts)):
        layers.append(
            {
                "layer": layer,
                "full_moe": True,
                "experts_run": 8,
                "selected_experts": experts,
                "routing_weights": [0.125] * 8,
            }
        )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "probe": "state_loop",
                "prompt_token": 42,
                "prompt_token_count": 1,
                "prompt_token_ids": [42],
                "generation_steps": 1,
                "steps": 1,
                "layers": 3,
                "start_layer": 1,
                "ctx_tokens": 4,
                "kv_format": "f16",
                "activation_format": "q8_k_compat",
                "residual_source": "injected_f32_replay",
                "residual_replay": {
                    "enabled": True,
                    "source": "f32_sidecar",
                    "values": 2,
                },
                "residual_dump": True,
                "residual_dump_count": 12,
                "delta_source": "real_attention_moe",
                "tokens": [
                    {
                        "step": 0,
                        "position": 0,
                        "input_token": 42,
                        "layers": layers,
                    }
                ],
                "layers_run": 2,
                "kv_appends": 2,
                "cache_readback_ok": True,
            }
        ),
        encoding="utf-8",
    )
    write_f32(run_dir / "step-0-layer-2-output.f32", final)


def populate_runs(experiment: Path, manifest: dict) -> None:
    for item in manifest["runs"]:
        write_run(
            experiment / item["directory"],
            final=[10.0, 20.0],
            layer2_experts=BASE_EXPERTS,
        )


def analyze(oracle: Path, baseline: Path, experiment: Path) -> subprocess.CompletedProcess[str]:
    return run_script(
        "analyze",
        "--oracle-dir",
        str(oracle),
        "--baseline-dir",
        str(baseline),
        "--experiment-dir",
        str(experiment),
    )


def set_nested(payload: dict, path: tuple[object, ...], value: object) -> None:
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_prepare_and_analyze_report_effective_gain_and_routing_transition(tmp_path):
    oracle, baseline, experiment = prepare_fixture(tmp_path)
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    runs = {float(item["scale"]): experiment / item["directory"] for item in manifest["runs"]}

    assert read_f32(runs[0.0] / "residual.f32") == pytest.approx([1.0, 2.0])
    assert read_f32(runs[1.0] / "residual.f32") == pytest.approx([1.25, 1.5])
    assert read_f32(runs[2.0] / "residual.f32") == pytest.approx([1.5, 1.0])

    write_run(runs[0.0], final=[10.0, 20.0], layer2_experts=BASE_EXPERTS)
    write_run(
        runs[1.0],
        final=[11.0, 20.0],
        layer2_experts=[11, 10, 12, 13, 14, 15, 16, 17],
    )
    write_run(
        runs[2.0],
        final=[14.0, 20.0],
        layer2_experts=[10, 11, 12, 13, 14, 15, 16, 18],
    )

    completed = analyze(oracle, baseline, experiment)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["schema"] == "qx-scaled-residual-replay-v1"
    rows = {float(row["scale"]): row for row in report["rows"]}
    assert rows[0.0]["final_vs_scale_zero"]["l2"] == 0.0
    assert rows[1.0]["effective_input_delta"]["l2"] == pytest.approx(math.sqrt(0.3125))
    assert rows[1.0]["requested_direction_fit"]["projected_scale"] == pytest.approx(1.0)
    assert rows[1.0]["requested_direction_fit"]["cosine"] == pytest.approx(1.0)
    assert rows[1.0]["requested_direction_fit"]["error_l2"] == 0.0
    assert rows[1.0]["suffix_response_gain_l2"] == pytest.approx(1.0 / math.sqrt(0.3125))
    assert rows[1.0]["routing_order_transition_layers"] == [2]
    assert rows[1.0]["routing_membership_transition_layers"] == []
    assert rows[2.0]["routing_membership_transition_layers"] == [2]
    assert report["routing"]["order_transition_scales"] == [1.0, 2.0]
    assert report["routing"]["membership_transition_scales"] == [2.0]
    assert report["verdict"] == "topk_membership_transition_observed"


def test_analyze_fails_closed_for_metadata_mismatch(tmp_path):
    oracle, baseline, experiment = prepare_fixture(tmp_path)
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    populate_runs(experiment, manifest)
    bad = experiment / manifest["runs"][1]["directory"] / "result.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["kv_format"] = "f32"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    completed = analyze(oracle, baseline, experiment)

    assert completed.returncode != 0
    assert "expected kv_format f16" in completed.stderr


@pytest.mark.parametrize(
    ("path", "bad_value", "expected_error"),
    [
        (("matrix", "layers"), True, "invalid matrix dimensions"),
        (("matrix", "start_layer"), 1.0, "invalid matrix dimensions"),
        (("matrix", "expected_count"), True, "invalid matrix dimensions"),
        (("matrix", "prompt_token"), True, "invalid execution matrix"),
        (("matrix", "ctx"), 4.0, "invalid execution matrix"),
        (("matrix", "seed"), False, "invalid execution matrix"),
    ],
)
def test_analyze_rejects_non_integer_manifest_metadata(
    tmp_path, path, bad_value, expected_error
):
    oracle, baseline, experiment = prepare_fixture(tmp_path)
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    populate_runs(experiment, manifest)
    set_nested(manifest, path, bad_value)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    completed = analyze(oracle, baseline, experiment)

    assert completed.returncode != 0
    assert expected_error in completed.stderr


def test_analyze_rejects_boolean_manifest_scale(tmp_path):
    oracle, baseline, experiment = prepare_fixture(tmp_path)
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    populate_runs(experiment, manifest)
    manifest["runs"][2]["scale"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    completed = analyze(oracle, baseline, experiment)

    assert completed.returncode != 0
    assert "manifest run is invalid" in completed.stderr


@pytest.mark.parametrize(
    ("path", "bad_value", "expected_error"),
    [
        (("start_layer",), True, "expected start_layer"),
        (("residual_replay", "values"), 2.0, "source or value count is invalid"),
        (("prompt_token",), True, "fixed one-token full-MoE replay matrix"),
        (("prompt_token_count",), 1.0, "fixed one-token full-MoE replay matrix"),
        (("prompt_token_ids", 0), True, "fixed one-token full-MoE replay matrix"),
        (("generation_steps",), True, "fixed one-token full-MoE replay matrix"),
        (("steps",), 1.0, "fixed one-token full-MoE replay matrix"),
        (("layers",), 3.0, "fixed one-token full-MoE replay matrix"),
        (("ctx_tokens",), 4.0, "fixed one-token full-MoE replay matrix"),
        (("residual_dump_count",), 12.0, "fixed one-token full-MoE replay matrix"),
        (("layers_run",), 2.0, "fixed one-token full-MoE replay matrix"),
        (("kv_appends",), 2.0, "fixed one-token full-MoE replay matrix"),
        (("tokens", 0, "step"), False, "token trace does not match"),
        (("tokens", 0, "position"), 0.0, "token trace does not match"),
        (("tokens", 0, "input_token"), True, "token trace does not match"),
        (("tokens", 0, "layers", 0, "layer"), True, "invalid layer trace row"),
        (("tokens", 0, "layers", 0, "experts_run"), 8.0, "expected full MoE top-8"),
        (("tokens", 0, "layers", 0, "selected_experts", 0), True, "invalid routing trace"),
        (("tokens", 0, "layers", 0, "selected_experts", 0), 1.0, "invalid routing trace"),
        (("tokens", 0, "layers", 0, "routing_weights", 0), True, "invalid routing trace"),
    ],
)
def test_analyze_rejects_non_integer_or_boolean_routing_metadata(
    tmp_path, path, bad_value, expected_error
):
    oracle, baseline, experiment = prepare_fixture(tmp_path)
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    populate_runs(experiment, manifest)
    result_path = experiment / manifest["runs"][1]["directory"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    set_nested(result, path, bad_value)
    result_path.write_text(json.dumps(result), encoding="utf-8")

    completed = analyze(oracle, baseline, experiment)

    assert completed.returncode != 0
    assert expected_error in completed.stderr


def test_analyze_fails_closed_when_trace_is_not_full_moe(tmp_path):
    oracle, baseline, experiment = prepare_fixture(tmp_path)
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    populate_runs(experiment, manifest)
    bad = experiment / manifest["runs"][1]["directory"] / "result.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["tokens"][0]["layers"][0]["full_moe"] = False
    bad.write_text(json.dumps(payload), encoding="utf-8")

    completed = analyze(oracle, baseline, experiment)

    assert completed.returncode != 0
    assert "expected full MoE top-8 trace" in completed.stderr


def test_analyze_fails_closed_for_non_object_manifest_json(tmp_path):
    oracle, baseline, experiment = prepare_fixture(tmp_path)
    (experiment / "manifest.json").write_text("[]", encoding="utf-8")

    completed = analyze(oracle, baseline, experiment)

    assert completed.returncode != 0
    assert "manifest root must be a JSON object" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_analyze_fails_closed_for_non_object_result_json(tmp_path):
    oracle, baseline, experiment = prepare_fixture(tmp_path)
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    populate_runs(experiment, manifest)
    bad = experiment / manifest["runs"][1]["directory"] / "result.json"
    bad.write_text("[]", encoding="utf-8")

    completed = analyze(oracle, baseline, experiment)

    assert completed.returncode != 0
    assert "result root must be a JSON object" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_analyze_fails_closed_for_tampered_residual(tmp_path):
    oracle, baseline, experiment = prepare_fixture(tmp_path)
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    populate_runs(experiment, manifest)
    write_f32(experiment / manifest["runs"][2]["directory"] / "residual.f32", [9.0, 9.0])

    completed = analyze(oracle, baseline, experiment)

    assert completed.returncode != 0
    assert "residual hash mismatch" in completed.stderr


@pytest.mark.parametrize("scales", ["0,1,1", "0,nan,1", "0", "1,2"])
def test_prepare_rejects_invalid_scale_grids(tmp_path, scales):
    oracle = tmp_path / "oracle"
    baseline = tmp_path / "baseline"
    write_f32(oracle / "layer-1.f32", [1.0, 2.0])
    write_f32(oracle / "l_out-2.f32", [10.0, 20.0])
    write_f32(baseline / "step-0-layer-0-output.f32", [1.25, 1.5])

    completed = run_script(
        "prepare",
        "--oracle-dir",
        str(oracle),
        "--baseline-dir",
        str(baseline),
        "--experiment-dir",
        str(tmp_path / "experiment"),
        f"--scales={scales}",
        "--layers",
        "3",
        "--start-layer",
        "1",
        "--expected-count",
        "2",
        "--kv-format",
        "f16",
        "--activation-format",
        "q8_k_compat",
    )

    assert completed.returncode != 0
