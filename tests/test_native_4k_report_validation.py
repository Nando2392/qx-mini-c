from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "native_4k_acceptance.py"
SPEC = importlib.util.spec_from_file_location("native_4k_report_validation", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def record(path: Path, relative_to: Path) -> dict:
    data = path.read_bytes()
    return {"path": path.relative_to(relative_to).as_posix(), "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def synthetic_report(tmp_path: Path) -> tuple[dict, Path, Path]:
    """Clearly synthetic files/report; this is never native acceptance evidence."""
    root, output = tmp_path / "repo", tmp_path / "repo" / "evidence"
    output.mkdir(parents=True)
    for index, relative in enumerate(runner.SOURCE_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-source-{index}".encode())
    exe, model, tokenizer = root / "build/qxqxf.exe", root / "models/model.qxf", root / "models/tokenizer.qxt"
    for path, data in ((exe, b"synthetic-exe"), (model, b"synthetic-model"), (tokenizer, b"synthetic-tokenizer")):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    stdout, stderr = output / "stdout.bin", output / "stderr.bin"
    stdout.write_bytes(b"synthetic stdout"); stderr.write_bytes(b"synthetic stderr")
    profile = {
        "requested_io_backend": "buffered", "effective_io_backend": "buffered",
        "requested_scratch_policy": "ephemeral", "effective_scratch_policy": "ephemeral",
        "requested_kernel_policy": "baseline", "effective_kernel_policy": "baseline",
        "requested_thread_policy": "serial", "effective_thread_policy": "serial",
        "requested_thread_count": 1, "effective_thread_count": 1, "sampled_steps": 2,
        "workers_used": 1, "scratch_peak_capacity_bytes": 0, "scratch_growth_events": 0,
        "temporary_blocks_decoded": 1, "temporary_floats_materialized": 1,
        "temporary_bytes_materialized": 4, "fused_final_head_dot_calls": 0,
        "baseline_final_head_dot_calls": 2, "final_head_q6_k_blocks": 2,
        "final_head_parallel_jobs": 0, "final_head_serial_jobs": 2,
        "final_head_fallback_jobs": 2, "full_logits_checksums": ["11", "22"],
    }
    raw = {"payload": {"prompt_token_count": 4095, "generated_token_ids": [7, 8],
        "generated_text": "synthetic", "stop_reason": "max_tokens",
        "timing": {"prefill_seconds": 1.0, "decode_seconds": .5}, "execution_profile": profile,
        "capacity_profile": {"executed_forward_steps": 4096, "prompt_forward_steps": 4095,
            "generated_input_forward_steps": 1, "first_eos_output_index": None}},
        "wall_seconds": 2.0, "sampled_peak_rss_bytes": 123456,
        "stdout": record(stdout, output), "stderr": record(stderr, output)}
    case = runner.validate_native_run(raw)
    sources = [record(root / relative, root) for relative in runner.SOURCE_FILES]
    inputs = {"git_revision": "1" * 40, "source_files": sources,
        "source_manifest_sha256": runner.sha256_bytes(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()),
        "native_executable": record(exe, root), "model_qxf": record(model, root),
        "tokenizer_qxt": record(tokenizer, root)}
    report = {"schema": runner.SCHEMA, "issue": 84, "status": "pass",
        "claim_scope": runner.CLAIM_SCOPE, "fixed_contract": copy.deepcopy(runner.FIXED_CONTRACT),
        "provenance": {"platform": {"system": "SyntheticOS", "release": "1", "machine": "x86_64",
            "processor": "synthetic-cpu", "python": "3.11.0", "psutil": "7.0.0"},
            "inputs_pre": inputs, "inputs_post": copy.deepcopy(inputs)},
        "reproduction": runner.reproduction_metadata(), "case": case,
        "gates": copy.deepcopy(runner.PASS_GATES)}
    return report, root, output


def reject(report: dict) -> None:
    with pytest.raises((TypeError, ValueError, KeyError)):
        runner.validate_report(report)


def mutate(report: dict, path: tuple, value: object) -> dict:
    changed = copy.deepcopy(report); target = changed
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    return changed


def test_strict_schema_and_real_artifacts_pass_separately(synthetic_report):
    report, root, output = synthetic_report
    assert runner.validate_report(report) is None
    assert runner.report_artifacts(report, root, output) is None


@pytest.mark.parametrize("field", ["schema", "issue", "status", "claim_scope", "fixed_contract", "provenance", "reproduction", "case", "gates"])
def test_all_top_level_fields_are_required(synthetic_report, field):
    report, _, _ = synthetic_report; del report[field]; reject(report)


@pytest.mark.parametrize(("path", "value"), [
    (("claim_scope",), ""), (("claim_scope",), "One 4K run"),
    (("reproduction",), {}), (("reproduction", "working_directory"), "."),
    (("reproduction", "run_command"), []), (("reproduction", "prepare_command"), ["python"]),
    (("provenance", "platform"), {}), (("provenance", "platform", "python"), ""),
    (("provenance", "platform", "python"), True), (("gates", "inputs_unchanged"), 1),
    (("fixed_contract", "threads"), True),
])
def test_claim_reproduction_platform_and_booleans_are_exact(synthetic_report, path, value):
    report, _, _ = synthetic_report; reject(mutate(report, path, value))


@pytest.mark.parametrize(("path", "value"), [
    (("provenance", "inputs_pre", "git_revision"), "a" * 39),
    (("provenance", "inputs_pre", "source_files"), []),
    (("provenance", "inputs_pre", "source_manifest_sha256"), "0" * 64),
    (("provenance", "inputs_pre", "model_qxf", "sha256"), "G" * 64),
    (("provenance", "inputs_pre", "tokenizer_qxt", "size_bytes"), True),
    (("provenance", "inputs_pre", "native_executable", "path"), ""),
])
def test_provenance_manifest_and_artifact_shape_fail_closed(synthetic_report, path, value):
    report, _, _ = synthetic_report
    changed = mutate(report, path, value)
    changed["provenance"]["inputs_post"] = copy.deepcopy(changed["provenance"]["inputs_pre"])
    reject(changed)


def test_self_consistent_forged_source_manifest_needs_disk_authentication(synthetic_report):
    report, root, output = synthetic_report
    for side in ("inputs_pre", "inputs_post"):
        records = report["provenance"][side]["source_files"]
        records[0]["sha256"] = "0" * 64
        report["provenance"][side]["source_manifest_sha256"] = runner.sha256_bytes(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode())
    runner.validate_report(report)  # offline consistency is not filesystem authenticity
    with pytest.raises(ValueError, match="source_files\\[0\\]"):
        runner.report_artifacts(report, root, output)


@pytest.mark.parametrize("bad_path", ["../stdout.bin", "/tmp/stdout.bin", "C:/Users/x/stdout.bin", r"C:\Users\x\stdout.bin", r"\\server\share\stdout.bin", "folder//stdout.bin", "folder/./stdout.bin"])
def test_artifact_paths_reject_traversal_absolute_unc_and_noncanonical(synthetic_report, bad_path):
    report, _, _ = synthetic_report; report["case"]["artifacts"]["stdout"]["path"] = bad_path; reject(report)


@pytest.mark.parametrize(("artifact", "corruption"), [("stdout", "missing"), ("stderr", "size"), ("stdout", "hash")])
def test_disk_verifier_rejects_missing_size_and_hash(synthetic_report, artifact, corruption):
    report, root, output = synthetic_report; path = output / f"{artifact}.bin"
    if corruption == "missing": path.unlink()
    elif corruption == "size": report["case"]["artifacts"][artifact]["size_bytes"] += 1
    else: report["case"]["artifacts"][artifact]["sha256"] = "0" * 64
    with pytest.raises(ValueError): runner.report_artifacts(report, root, output)


def test_disk_verifier_rejects_self_consistent_report_after_file_corruption(synthetic_report):
    report, root, output = synthetic_report
    (output / "stdout.bin").write_bytes(b"corrupted after report construction")
    with pytest.raises(ValueError, match="stdout"):
        runner.report_artifacts(report, root, output)


def test_disk_verifier_checks_model_and_tokenizer_actual_bytes(synthetic_report):
    report, root, output = synthetic_report
    (root / report["provenance"]["inputs_pre"]["model_qxf"]["path"]).write_bytes(b"changed model")
    with pytest.raises(ValueError, match="model_qxf"):
        runner.report_artifacts(report, root, output)


def test_disk_verifier_requires_confined_root_and_output(synthetic_report, tmp_path):
    report, root, _ = synthetic_report
    with pytest.raises(ValueError, match="output_dir"):
        runner.report_artifacts(report, root, tmp_path / "outside")
