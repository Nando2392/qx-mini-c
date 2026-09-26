"""Deterministic unit tests for the Issue 86 acceptance harness.

All QXF metadata, native payloads, processes, hashes, and measurements in this
module are synthetic test data.  They are never Issue 86 acceptance proof.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "scripts" / "expert_cache_acceptance.py"
LOGIT_HELPER_PATH = REPO_ROOT / "scripts" / "expert_cache_logit_bytes.py"
LOGIT_SPEC = importlib.util.spec_from_file_location("expert_cache_logit_bytes", LOGIT_HELPER_PATH)
assert LOGIT_SPEC is not None and LOGIT_SPEC.loader is not None
logit_helper = importlib.util.module_from_spec(LOGIT_SPEC)
sys.modules[LOGIT_SPEC.name] = logit_helper
LOGIT_SPEC.loader.exec_module(logit_helper)
SPEC = importlib.util.spec_from_file_location("issue86_expert_cache_acceptance", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
harness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = harness
SPEC.loader.exec_module(harness)

EXPECTED_BUDGET = 616_464_384
EXPERT_SLICE_SIZES = {"gate": 401_344, "up": 802_688, "down": 401_344}


def synthetic_metadata() -> Any:
    """Return deterministic directory metadata, not data from a real model."""
    tensors = {}
    for layer in range(harness.LAYERS):
        for kind, slice_bytes in EXPERT_SLICE_SIZES.items():
            name = f"blk.{layer}.ffn_{kind}_exps.weight"
            tensors[name] = harness.QxfTensor(name, 0, slice_bytes * 128)
    return harness.QxfMetadata(1, harness.LAYERS, 128, harness.TOP_K, tensors)


def synthetic_raw(policy: str = "resident-packed", *, hits: int = 4) -> dict[str, Any]:
    """Return one internally consistent synthetic native run payload."""
    enabled = policy == "resident-packed"
    requests = harness.EXPECTED_RESIDENT_REQUESTS if enabled else 0
    misses = requests - hits if enabled else 0
    budget = EXPECTED_BUDGET if enabled else 0
    cache = {
        "policy": policy,
        "enabled": enabled,
        "budget_bytes": budget,
        "requests": requests,
        "hits": hits if enabled else 0,
        "misses": misses,
        "loads": misses,
        "evictions": 0,
        "current_resident_packed_bytes": 1_000 if enabled else 0,
        "peak_resident_packed_bytes": 2_000 if enabled else 0,
        "buffered_bytes_read": 2_000 if enabled else 0,
        "buffered_bytes_avoided": 4_000 if enabled and hits else 0,
    }
    generation = [
        {
            "step": step,
            "phase": "generate",
            "selected_token": token,
            "final_head": {
                "logits_checksum": checksum,
                "full_vocabulary": True,
                "logits_computed": 151_936,
            },
        }
        for step, (token, checksum) in enumerate(zip(harness.EXPECTED_F32_TOKENS, (11, 12)))
    ]
    return {
        "returncode": 0,
        "timed_out": False,
        "wall_seconds": 1.25,
        "sampled_peak_rss_bytes": 123_456,
        "stdout": {"path": "synthetic/stdout.json"},
        "stderr": {"path": "synthetic/stderr.log"},
        "payload": {
            "activation_format": "f32",
            "io_backend": "buffered",
            "layers": harness.LAYERS,
            "ctx_tokens": harness.CTX,
            "prompt_token_count": 1,
            "steps": 2,
            "tokens": generation,
            "expert_cache_profile": cache,
        },
    }


def synthetic_compact_run() -> dict[str, Any]:
    return {
        "wall_seconds": 1.0,
        "sampled_peak_rss_bytes": 100,
        "generated_token_ids": list(harness.EXPECTED_F32_TOKENS),
        "full_logits_checksums": ["11", "12"],
    }


def test_budget_derives_exact_48_layer_top8_gate_up_down_working_set() -> None:
    budget = harness.derive_budget(synthetic_metadata(), available_bytes=8 * 1024**3)

    assert budget["budget_bytes"] == EXPECTED_BUDGET
    assert budget["feasible"] is True
    assert len(budget["per_layer"]) == 48
    assert all(item["top_k_working_set_bytes"] == 8 * 1_605_376 for item in budget["per_layer"])
    assert all(item["one_expert_gate_up_down_bytes"] == 1_605_376 for item in budget["per_layer"])


def test_feasible_budget_does_not_assume_lru_hits_are_guaranteed() -> None:
    """A full-scan LRU can thrash; feasibility alone is not positive-hit proof."""
    budget = harness.derive_budget(synthetic_metadata(), available_bytes=8 * 1024**3)
    raw = synthetic_raw(hits=0)
    cache = raw["payload"]["expert_cache_profile"]
    cache["requests"] = harness.EXPECTED_RESIDENT_REQUESTS
    cache["misses"] = harness.EXPECTED_RESIDENT_REQUESTS
    cache["loads"] = harness.EXPECTED_RESIDENT_REQUESTS

    assert budget["feasible"] is True
    with pytest.raises(harness.AcceptanceError, match="positive repeated-expert proof"):
        harness.compact_run(raw, "f32", "resident-packed", EXPECTED_BUDGET)


@pytest.mark.parametrize("bad_value", [True, False, 1.0, "1", None])
def test_cache_integer_counters_reject_non_exact_int_types(bad_value: object) -> None:
    raw = synthetic_raw()
    raw["payload"]["expert_cache_profile"]["hits"] = bad_value

    with pytest.raises(harness.AcceptanceError, match=r"cache\.hits must be an integer"):
        harness.compact_run(raw, "f32", "resident-packed", EXPECTED_BUDGET)


@pytest.mark.parametrize("bad_value", [1, 0, "true", None])
def test_cache_enabled_requires_json_boolean_identity(bad_value: object) -> None:
    raw = synthetic_raw()
    raw["payload"]["expert_cache_profile"]["enabled"] = bad_value

    with pytest.raises(harness.AcceptanceError, match="requested/effective policy mismatch"):
        harness.compact_run(raw, "f32", "resident-packed", EXPECTED_BUDGET)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_wall_measurements_fail_closed(bad_value: float) -> None:
    raw = synthetic_raw()
    raw["wall_seconds"] = bad_value

    with pytest.raises(harness.AcceptanceError, match="finite and positive"):
        harness.compact_run(raw, "f32", "resident-packed", EXPECTED_BUDGET)


@pytest.mark.parametrize("missing_index", [0, 2])
def test_summary_missing_first_or_later_measurement_field_is_acceptance_error(missing_index: int) -> None:
    runs = [synthetic_compact_run() for _ in range(3)]
    runs[missing_index].pop("sampled_peak_rss_bytes")

    with pytest.raises(harness.AcceptanceError, match="missing required field.*sampled_peak_rss_bytes"):
        harness.summary(runs)


def test_qxf_zero_architecture_widths_are_rejected(tmp_path: Path) -> None:
    """A model-shaped header with zero hidden/FFN widths must not pass preflight."""
    raw = bytearray(272)
    raw[:4] = b"QXF1"
    struct.pack_into("<4I", raw, 4, 1, 108, 176, 0)
    struct.pack_into("<4Q", raw, 24, 272, 272, 272, 0)
    manifest = [0] * 13
    manifest[2] = harness.LAYERS
    manifest[10] = 128
    manifest[11] = harness.TOP_K
    struct.pack_into("<13I", raw, 56, *manifest)
    path = tmp_path / "synthetic-zero-widths.qxf"
    path.write_bytes(raw)

    with pytest.raises(harness.AcceptanceError, match="width|manifest"):
        harness.parse_qxf(path)


def test_reproduction_keeps_raw_output_under_build_issue86_prefix() -> None:
    parent = harness.reproduction(EXPECTED_BUDGET)["parent_command"]
    output = parent[parent.index("--output-dir") + 1].replace("\\", "/")

    assert output.startswith("build/issue86-")
    assert not output.startswith("wiki/")


def test_main_rejects_raw_output_outside_build_issue86_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private or large raw artifacts must not be written to publication paths."""
    output = tmp_path / "wiki" / "evidence" / "issue-86-test-data"
    test_input = Path(__file__).resolve()
    freeze_reached = False

    monkeypatch.setattr(sys, "argv", [
        str(HARNESS_PATH), "--driver-exe", str(test_input), "--model", str(test_input),
        "--tokenizer", str(test_input), "--output-dir", str(output),
    ])
    monkeypatch.setattr(harness, "portable_path", lambda path, root: "wiki/evidence/issue-86-test-data")
    monkeypatch.setattr(harness, "parse_qxf", lambda path: synthetic_metadata())
    monkeypatch.setattr(harness.psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3))
    monkeypatch.setattr(harness, "atomic_json", lambda path, value: None)

    def observe_freeze(*args: Any) -> dict[str, Any]:
        nonlocal freeze_reached
        freeze_reached = True
        raise harness.AcceptanceError("synthetic stop")

    monkeypatch.setattr(harness, "freeze", observe_freeze)

    assert harness.main() == 2
    assert freeze_reached is False, "invalid raw-output roots must fail before hashing or native work"


def test_equivalence_never_upgrades_checksums_to_full_byte_claim(tmp_path: Path) -> None:
    cells = []
    for activation in harness.ACTIVATIONS:
        for policy in harness.POLICIES:
            cells.append({
                "activation": activation,
                "policy": policy,
                "warmups": [synthetic_compact_run()],
                "measured": [synthetic_compact_run() for _ in range(3)],
                "evidence_root": str(tmp_path),
            })

    with pytest.raises(harness.AcceptanceError, match="missing full-logit byte artifacts"):
        harness.equivalence(cells)


def test_run_one_samples_parent_and_descendant_rss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeNativeProcess:
        pid = 9001
        returncode = 0

        def __init__(self, command: list[str], stdout: Any, stderr: Any) -> None:
            del command, stderr
            stdout.write(json.dumps({"synthetic": True}).encode("utf-8"))
            self.polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls == 1 else 0

    class FakePsutilProcess:
        def __init__(self, pid: int, rss: int = 10) -> None:
            self.pid = pid
            self.rss = rss

        def memory_info(self) -> Any:
            return SimpleNamespace(rss=self.rss)

        def children(self, recursive: bool = False) -> list[Any]:
            assert recursive is True
            return [FakePsutilProcess(9002, 20), FakePsutilProcess(9003, 30)]

    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(harness.subprocess, "Popen", FakeNativeProcess)
    monkeypatch.setattr(harness.psutil, "Process", FakePsutilProcess)
    monkeypatch.setattr(harness.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(harness.time, "sleep", lambda _seconds: None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = harness.run_one(["synthetic-native-command"], run_dir, parent_deadline=100.0)

    assert result["sampled_peak_rss_bytes"] == 60
    assert result["payload"] == {"synthetic": True}


def test_timeout_cleanup_kills_parent_and_recursive_descendants(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []

    class FakePsutilProcess:
        def __init__(self, pid: int, children: list[Any] | None = None) -> None:
            self.pid = pid
            self._children = children or []

        def children(self, recursive: bool = False) -> list[Any]:
            assert recursive is True
            return self._children

        def kill(self) -> None:
            killed.append(self.pid)

    descendants = [FakePsutilProcess(12), FakePsutilProcess(13)]
    psutil_parent = FakePsutilProcess(11, descendants)

    class FakePopen:
        pid = 11

        def kill(self) -> None:
            killed.append(self.pid)

        def wait(self, timeout: float) -> int:
            assert timeout == 10
            return 0

    waited: list[list[int]] = []
    monkeypatch.setattr(harness.psutil, "Process", lambda pid: psutil_parent)
    monkeypatch.setattr(
        harness.psutil,
        "wait_procs",
        lambda processes, timeout: (waited.append([p.pid for p in processes]) or (processes, [])),
    )

    harness.stop_tree(FakePopen())

    assert {12, 13}.issubset(killed)
    assert 11 in killed
    assert waited == [[12, 13, 11]]


def test_main_executes_exactly_warmup_plus_three_runs_for_each_of_four_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count mocked native launches; none of these calls execute a real model."""
    output = tmp_path / "issue86-matrix-test-data"
    test_input = Path(__file__).resolve()
    calls: list[list[str]] = []
    freezes: list[str] = []

    monkeypatch.setattr(sys, "argv", [
        str(HARNESS_PATH), "--driver-exe", str(test_input), "--model", str(test_input),
        "--tokenizer", str(test_input), "--output-dir", str(output),
    ])
    monkeypatch.setattr(harness, "portable_path", lambda path, root: "synthetic/test-data")
    monkeypatch.setattr(harness, "parse_qxf", lambda path: synthetic_metadata())
    monkeypatch.setattr(harness.psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3))
    monkeypatch.setattr(harness, "atomic_json", lambda path, value: None)
    monkeypatch.setattr(harness, "artifact", lambda path, root: {"path": "synthetic/test-data", "size_bytes": 1, "sha256": "0" * 64})
    monkeypatch.setattr(harness, "sha256_file", lambda path: "0" * 64)

    def fake_freeze(*args: Any) -> dict[str, Any]:
        freezes.append("freeze")
        return {"synthetic": "frozen-test-data"}

    def fake_run(command: list[str], run_dir: Path, deadline: float) -> dict[str, Any]:
        del run_dir, deadline
        calls.append(command)
        return {"synthetic": True}

    monkeypatch.setattr(harness, "freeze", fake_freeze)
    monkeypatch.setattr(harness, "run_one", fake_run)
    monkeypatch.setattr(
        harness,
        "compact_run",
        lambda raw, activation, policy, budget, capture_dir, evidence_root: {
            **synthetic_compact_run(),
            "full_logit_artifacts": [
                {
                    "step": step,
                    "path": f"synthetic/test-data/logits-{step}.f32le",
                    "size_bytes": 4 * 151_936,
                    "sha256": "0" * 64,
                    "argmax_token_id": token,
                    "vocab_size": 151_936,
                    "all_finite": True,
                }
                for step, token in enumerate(harness.EXPECTED_F32_TOKENS)
            ],
        },
    )
    monkeypatch.setattr(harness, "equivalence", lambda cells: {"status": "synthetic-test-data"})

    assert harness.main() == 0
    assert len(calls) == 16
    tuples = Counter((
        command[command.index("--activation") + 1],
        command[command.index("--expert-cache-policy") + 1],
    ) for command in calls)
    assert tuples == Counter({(activation, policy): 4 for activation in harness.ACTIVATIONS for policy in harness.POLICIES})
    assert len(freezes) == 2
    assert (output / "prompt.txt").read_text(encoding="utf-8") == harness.PROMPT_TEXT


def test_parent_deadline_starts_before_pre_run_hashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "issue86-deadline-test-data"
    test_input = Path(__file__).resolve()
    events: list[str] = []

    monkeypatch.setattr(sys, "argv", [
        str(HARNESS_PATH), "--driver-exe", str(test_input), "--model", str(test_input),
        "--tokenizer", str(test_input), "--output-dir", str(output),
    ])
    monkeypatch.setattr(harness, "portable_path", lambda path, root: "synthetic/test-data")
    monkeypatch.setattr(harness, "parse_qxf", lambda path: synthetic_metadata())
    monkeypatch.setattr(harness.psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3))
    monkeypatch.setattr(harness, "atomic_json", lambda path, value: None)
    monkeypatch.setattr(harness.time, "monotonic", lambda: events.append("deadline") or 10.0)

    def stop_at_freeze(*args: Any) -> dict[str, Any]:
        events.append("freeze")
        raise harness.AcceptanceError("synthetic stop after observing ordering")

    monkeypatch.setattr(harness, "freeze", stop_at_freeze)

    assert harness.main() == 2
    assert events[0] == "deadline", "the 2100-second parent budget must include pre-run hashing"
