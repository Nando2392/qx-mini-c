"""Unit fixtures are synthetic; these tests never launch a model matrix."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expert_cache_acceptance as base


def tooling():
    return importlib.import_module("moe_pool_acceptance")


@pytest.mark.parametrize("phase", ["initialize", "descendants", "sample"])
def test_real_root_exit_at_psutil_operation_uses_one_final_query(tmp_path, monkeypatch, phase):
    """Synchronize exit before the real psutil call, not a fabricated exception."""
    h = tooling()
    release = tmp_path / "release"
    launched = []
    original_popen = subprocess.Popen
    query = h.final_peak_working_set
    peaks = []

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launched.append(process)
        return process

    name = {"initialize": "__init__", "descendants": "children", "sample": "memory_info"}[phase]
    operation = getattr(base.psutil.Process, name)
    synchronized = []

    def exit_before_operation(process, *args, **kwargs):
        if not synchronized:
            synchronized.append(True)
            release.write_text("exit")
            assert launched[0].wait(timeout=10) == 0
        return operation(process, *args, **kwargs)

    def final_query(process):
        peak = query(process)
        peaks.append(peak)
        return peak

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(base.psutil.Process, name, exit_before_operation)
    monkeypatch.setattr(h, "final_peak_working_set", final_query)
    command = [sys._base_executable, "-c",
               "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
               "exec('while not p.exists(): time.sleep(.01)'); print('{}')", str(release)]
    record = h.run_one(command, tmp_path, ROOT)
    assert record["sampled_peak_rss_bytes"] == 0
    assert len(peaks) == 1 and peaks[0] > 0
    assert record["final_peak_working_set_bytes"] == peaks[0]
    assert record["rss_exit_race_recovered"] is True
    assert record["rss_scope"] == "root_process_working_set"
    assert record["hard_os_memory_limit"] is False
    assert record["lifetime_tree_guarantee"] is False


@pytest.mark.parametrize("outcome", ["unknown_pid", "missing_pid", "access_denied", "nonzero_exit"])
def test_exited_root_does_not_recover_unproven_enumeration_fault(tmp_path, monkeypatch, outcome):
    import json
    h = tooling()
    original_popen = subprocess.Popen
    launched = []
    release = tmp_path / "release"
    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launched.append(process)
        return process
    def children(process, **kwargs):
        release.write_text("exit")
        assert launched[0].wait(timeout=10) == (7 if outcome == "nonzero_exit" else 0)
        pid = None if outcome == "missing_pid" else process.pid
        if outcome == "unknown_pid":
            pid += 1
        error = base.psutil.AccessDenied if outcome == "access_denied" else base.psutil.NoSuchProcess
        raise error(pid, msg="unproven enumeration fault")
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(base.psutil.Process, "children", children)
    monkeypatch.setattr(h, "final_peak_working_set", lambda _: pytest.fail("must not query unproven root"))
    command = [sys._base_executable, "-c",
               "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
               "exec('while not p.exists(): time.sleep(.01)'); print('{}'); "
               f"sys.exit({7 if outcome == 'nonzero_exit' else 0})", str(release)]
    with pytest.raises(base.AcceptanceError, match="RSS monitoring failed"):
        h.run_one(command, tmp_path, ROOT)
    record = json.loads((tmp_path / "process.json").read_text())
    assert record["rss_monitoring_error"]["phase"] == "descendants"
    assert "final_peak_working_set_bytes" not in record


@pytest.mark.parametrize("over", [False, True])
def test_real_final_peak_cap_boundary(tmp_path, monkeypatch, over):
    import json
    h = tooling()
    original_popen = subprocess.Popen
    query = h.final_peak_working_set
    queries = []
    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        assert process.wait(timeout=10) == 0
        return process
    def final_query(process):
        peak = query(process)
        queries.append(peak)
        monkeypatch.setattr(h, "MAX_RSS_BYTES", peak - int(over))
        return peak
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(h, "final_peak_working_set", final_query)
    command = [sys._base_executable, "-c", "print('{}')"]
    if over:
        with pytest.raises(base.AcceptanceError, match="native process failed"):
            h.run_one(command, tmp_path, ROOT)
        record = json.loads((tmp_path / "process.json").read_text())
    else:
        record = h.run_one(command, tmp_path, ROOT)
    assert len(queries) == 1 and queries[0] > 0
    assert record["sampled_peak_rss_bytes"] == 0
    assert record["final_peak_working_set_bytes"] == queries[0]
    assert record["rss_limit_exceeded"] is over
    assert record["returncode"] == 0


def test_no_such_process_outside_root_operation_is_not_recovered(tmp_path, monkeypatch):
    h = tooling()
    release = tmp_path / "release"
    original = subprocess.Popen
    launched = []
    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        launched.append(process)
        return process
    def sleep(_):
        release.write_text("exit")
        assert launched[0].wait(timeout=10) == 0
        raise base.psutil.NoSuchProcess(launched[0].pid, msg="not a root PID operation")
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(h.time, "sleep", sleep)
    monkeypatch.setattr(h, "final_peak_working_set", lambda _: pytest.fail("only root PID operations may recover"))
    command = [sys._base_executable, "-c",
               "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
               "exec('while not p.exists(): time.sleep(.01)'); print('{}')", str(release)]
    with pytest.raises(base.AcceptanceError, match="RSS monitoring failed"):
        h.run_one(command, tmp_path, ROOT)


def test_default_command_explicitly_preserves_serial_one():
    h = tooling()
    args = [Path(x) for x in ("exe", "model", "tokenizer", "prompt", "dump")]
    cmd = h.command_for(*args, "serial", "none", 100)
    assert "--thread-policy" in cmd
    assert cmd[cmd.index("--thread-policy") + 1] == "serial"
    assert cmd[cmd.index("--threads") + 1] == "1"
    assert cmd[cmd.index("--generate") + 1] == "2"


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    from moe_pool_build import build_driver

    out = tmp_path_factory.mktemp("isolated-driver")
    exe = build_driver(out, ROOT / "tests/expert_cache_acceptance_driver.c")
    return exe, out


def test_driver_fixture_builds_in_isolation_without_checkout_artifacts(tmp_path, monkeypatch):
    import moe_pool_build

    checkout = tmp_path / "clean-checkout"
    checkout.mkdir()
    out = tmp_path / "isolated-driver"
    out.mkdir()
    executable = out / "moe_pool_runtime_driver.exe"
    calls = []

    class TempFactory:
        def mktemp(self, name):
            assert name == "isolated-driver"
            return out

    def build(output_dir, source):
        calls.append((output_dir, source))
        return executable

    monkeypatch.setattr(sys.modules[__name__], "ROOT", checkout)
    monkeypatch.setattr(moe_pool_build, "build_driver", build)
    assert driver.__wrapped__(TempFactory()) == (executable, out)
    assert calls == [(out, checkout / "tests/expert_cache_acceptance_driver.c")]
    assert not (checkout / "build").exists()


def synthetic_raw():
    from test_expert_cache_acceptance import synthetic_raw as cache_raw
    raw = cache_raw("none")
    raw["final_peak_working_set_bytes"] = raw["sampled_peak_rss_bytes"]
    raw["payload"]["moe_pool_profile"] = {"workers": 2, "jobs": 10}
    return raw


@pytest.mark.parametrize("profile", [None, {}, {"workers": 0, "jobs": 10},
                                     {"workers": 2, "jobs": 0}, {"workers": True, "jobs": 10}])
def test_pool_requires_actual_positive_workers_and_jobs(profile):
    h = tooling()
    raw = synthetic_raw()
    raw["payload"]["moe_pool_profile"] = profile
    with pytest.raises(base.AcceptanceError):
        h.compact_run(raw, "moe-pool", "none", 100)


def test_matrix_is_twelve_fixed_f32_runs():
    h = tooling()
    plan = h.matrix_plan()
    assert len(plan) == 12
    assert {(t, p) for t, p, _, _ in plan} == {(t, p) for t in ("serial", "moe-pool") for p in base.POLICIES}


def test_serial_legacy_thread_profile_needs_no_new_pool_fields():
    from test_expert_cache_acceptance import synthetic_raw as cache_raw
    raw = cache_raw("none")
    raw["final_peak_working_set_bytes"] = raw["sampled_peak_rss_bytes"]
    raw["payload"]["thread_profile"] = {"policy": "serial", "workers_used": 1,
                                        "parallel_jobs": 0}
    result = tooling().compact_run(raw, "serial", "none", 100)
    assert result["moe_pool_profile"] == {"workers": 0, "jobs": 0}


@pytest.mark.parametrize("profile", [None, {}, {"policy": "pool", "workers_used": 2,
                                               "parallel_jobs": 2}])
def test_serial_cannot_silently_accept_another_thread_policy(profile):
    from test_expert_cache_acceptance import synthetic_raw as cache_raw
    raw = cache_raw("none")
    raw["final_peak_working_set_bytes"] = raw["sampled_peak_rss_bytes"]
    raw["payload"]["thread_profile"] = profile
    with pytest.raises(base.AcceptanceError):
        tooling().compact_run(raw, "serial", "none", 100)


def test_existing_output_is_not_overwritten(tmp_path):
    marker = tmp_path / "failure.json"
    marker.write_text("untouched")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/moe_pool_acceptance.py"),
                             "--driver-exe", str(Path(__file__).resolve()),
                             "--model", str(Path(__file__).resolve()),
                             "--tokenizer", str(Path(__file__).resolve()),
                             "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 2
    assert marker.read_text() == "untouched"


def test_process_logs_have_argv_pid_cwd_timestamps(tmp_path, monkeypatch):
    h = tooling()
    release = tmp_path / "release"
    original_popen = subprocess.Popen
    launched = []

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launched.append(process)
        return process

    def finish_between_samples(_):
        release.write_text("exit")
        assert launched[0].wait(timeout=10) == 0

    cmd = [sys._base_executable, "-c", "import json,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
           "exec('while not p.exists(): time.sleep(.01)'); print(json.dumps({'synthetic': True}))",
           str(release)]
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(h.time, "sleep", finish_between_samples)
    record = h.run_one(cmd, tmp_path, ROOT)
    assert record["rss_monitoring_error"] is None
    assert record["sampled_peak_rss_bytes"] > 0
    assert record["argv"] == cmd
    assert record["pid"] > 0
    assert record["cwd"] == str(ROOT)
    assert record["started_utc"] <= record["finished_utc"]
    assert record["timeout_seconds"] == 300


def test_equivalence_compares_every_cell_against_serial_bytes(tmp_path):
    import struct
    from expert_cache_logit_bytes import collect_generated_logits
    h = tooling()
    cells = []
    for thread in ("serial", "moe-pool"):
        for policy in base.POLICIES:
            capture = tmp_path / (thread + policy)
            capture.mkdir()
            steps = []
            for step, token in enumerate(base.EXPECTED_F32_TOKENS):
                raw = bytearray(base.VOCAB_SIZE * 4)
                struct.pack_into("<f", raw, token * 4, 1.0)
                (capture / f"step-{step}-logits.f32").write_bytes(raw)
                steps.append({"step": step, "phase": "generate", "selected_token": token})
            run = {"generated_token_ids": base.EXPECTED_F32_TOKENS,
                   "full_logits_checksums": ["11", "12"],
                   "full_logit_artifacts": collect_generated_logits(capture, steps, base.VOCAB_SIZE, tmp_path)}
            cells.append({"thread_policy": thread, "activation": "f32", "policy": policy,
                          "warmups": [run], "measured": [run, run], "evidence_root": str(tmp_path)})
    validator = h.equivalence
    result = validator(cells)
    assert result["runs_compared"] == 12
    assert result["none_runs_compared"] == 6
    assert result["resident_runs_compared"] == 6
    assert result["full_logit_bytes_equal"] is True
    original = cells[-1]["measured"][-1]
    cells[-1]["measured"][-1] = {**original, "full_logits_checksums": ["99", "12"]}
    with pytest.raises(base.AcceptanceError, match="exact tokens/checksums differ"):
        validator(cells)
    cells[-1]["measured"][-1] = original

    # Keep tokens, checksums and artifact provenance valid: only raw bytes differ.
    path = capture / "step-0-logits.f32"
    changed = bytearray(path.read_bytes())
    other_token = (base.EXPECTED_F32_TOKENS[0] + 1) % base.VOCAB_SIZE
    struct.pack_into("<f", changed, other_token * 4, 0.5)
    path.write_bytes(changed)
    original["full_logit_artifacts"] = collect_generated_logits(
        capture, steps, base.VOCAB_SIZE, tmp_path)
    with pytest.raises(base.AcceptanceError, match="full logit bytes differ"):
        validator(cells)


@pytest.mark.parametrize("wall", ["timeout", "rss"])
def test_process_wall_kills_real_child_and_records_failure(tmp_path, monkeypatch, wall):
    import json
    h = tooling()
    if wall == "rss":
        monkeypatch.setattr(h, "MAX_RSS_BYTES", 1)
    else:
        calls = iter([0.0])
        monkeypatch.setattr(h.time, "monotonic", lambda: next(calls, 301.0))
    with pytest.raises(base.AcceptanceError, match="native process failed"):
        h.run_one([sys._base_executable, "-c", "import time; time.sleep(30)"], tmp_path, ROOT)
    record = json.loads((tmp_path / "process.json").read_text())
    assert record["timed_out" if wall == "timeout" else "rss_limit_exceeded"] is True
    assert record["returncode"] != 0
    assert not base.psutil.pid_exists(record["pid"])


@pytest.mark.parametrize("root_exits", [False, True])
def test_observed_real_child_violates_root_driver_contract_and_is_cleaned(tmp_path, monkeypatch, root_exits):
    """Intentional change: threads-only driver rejects children, not tree RSS."""
    import json
    import time
    h = tooling()
    child_pid_path = tmp_path / "child.pid"
    original_children = base.psutil.Process.children
    original_popen = subprocess.Popen
    launched = []
    release = tmp_path / "release"

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launched.append(process)
        return process

    command = [sys._base_executable, "-c",
               "import pathlib,subprocess,sys,time; "
               "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
               "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); p=pathlib.Path(sys.argv[2]); "
               "exec('while not p.exists(): time.sleep(.01)'); print('{}')", str(child_pid_path), str(release)]

    def children(process, **kwargs):
        deadline = time.monotonic() + 10
        while not child_pid_path.exists():
            assert time.monotonic() < deadline, "child did not start"
            time.sleep(.01)
        found = original_children(process, **kwargs)
        if root_exits and found:
            release.write_text("exit")
            assert launched[0].wait(timeout=10) == 0
        return found

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(base.psutil.Process, "children", children)
    try:
        with pytest.raises(base.AcceptanceError, match="forbidden child"):
            h.run_one(command, tmp_path, ROOT)
        record = json.loads((tmp_path / "process.json").read_text())
        assert record["forbidden_child_pids"] == [int(child_pid_path.read_text())]
        assert record["rss_monitoring_error"]["phase"] == "descendants"
        assert record["sampled_peak_rss_bytes"] == 0
        assert not base.psutil.pid_exists(record["pid"])
        assert not base.psutil.pid_exists(int(child_pid_path.read_text()))
    finally:
        if child_pid_path.exists():
            try:
                child = base.psutil.Process(int(child_pid_path.read_text()))
                child.kill()
                child.wait(timeout=10)
            except base.psutil.NoSuchProcess:
                pass

@pytest.mark.parametrize("error_type", [base.psutil.AccessDenied, base.psutil.NoSuchProcess])
def test_initial_rss_tracking_failure_is_durable_and_kills_child(tmp_path, monkeypatch, error_type):
    import json
    h = tooling()
    original = base.psutil.Process.__init__
    attempted = []

    def initialize(process, pid=None):
        if not attempted:
            attempted.append(pid)
            raise error_type(pid, msg="injected initial tracking failure")
        original(process, pid)

    monkeypatch.setattr(base.psutil.Process, "__init__", initialize)
    with pytest.raises(base.AcceptanceError, match="RSS monitoring failed"):
        h.run_one([sys._base_executable, "-c", "import time; time.sleep(.3); print('{}')"], tmp_path, ROOT)
    record = json.loads((tmp_path / "process.json").read_text())
    assert record["rss_monitoring_error"]["type"] == error_type.__name__
    assert record["rss_monitoring_error"]["phase"] == "initialize"
    assert record["rss_monitoring_error"]["pid"] == attempted[0] == record["pid"]
    assert record["sampled_peak_rss_bytes"] == 0
    assert record["stdout"]["path"]
    assert record["stderr"]["path"]
    assert not base.psutil.pid_exists(record["pid"])


@pytest.mark.parametrize("outcome", ["pass", "query_failure", "peak_over_limit"])
def test_rss_exit_race_accepts_only_proven_final_lifetime_peak(tmp_path, monkeypatch, outcome):
    """Old rejection protected an unknown tail; a retained-handle peak proves it.

    Use the actual interpreter, not the venv launcher (which has a descendant).
    Only the exit timing is injected; the final Win32 query must be real.
    """
    import json
    h = tooling()
    release = tmp_path / "release"
    original_popen = subprocess.Popen
    original_memory = base.psutil.Process.memory_info
    launched = []
    samples = []

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launched.append(process)
        return process

    def memory_info(process):
        if process.pid != launched[0].pid:
            return original_memory(process)
        if samples:
            release.write_text("exit")
            assert launched[0].wait(timeout=10) == 0
            if outcome == "query_failure":
                # Exercise real Win32 failure, not a fabricated peak/result.
                launched[0]._handle.Close()
            elif outcome == "peak_over_limit":
                monkeypatch.setattr(h, "MAX_RSS_BYTES", samples[0] + 16 * 1024**2)
        result = original_memory(process)
        samples.append(result.rss)
        return result

    command = [sys._base_executable, "-c",
               "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
               "exec('while not p.exists(): time.sleep(.01)'); "
               "x=bytearray(32*1024*1024); x[::4096]=bytes([1])*len(x[::4096]); print('{}')",
               str(release)]
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(base.psutil.Process, "memory_info", memory_info)
    try:
        if outcome == "pass":
            record = h.run_one(command, tmp_path, ROOT)
        else:
            message = "RSS monitoring failed" if outcome == "query_failure" else "native process failed"
            with pytest.raises(base.AcceptanceError, match=message):
                h.run_one(command, tmp_path, ROOT)
            record = json.loads((tmp_path / "process.json").read_text())
            assert record["returncode"] == 0
            if outcome == "query_failure":
                assert record["rss_monitoring_error"]["phase"] == "final_peak"
                assert record["rss_monitoring_error"]["type"] == "OSError"
            else:
                assert record["final_peak_working_set_bytes"] > h.MAX_RSS_BYTES
                assert record["rss_limit_exceeded"] is True
            return
        assert samples[0] > 0
        assert record["final_peak_working_set_bytes"] >= 32 * 1024**2
        assert record["sampled_peak_rss_bytes"] == max(samples)
        assert record["sampled_peak_rss_bytes"] < record["final_peak_working_set_bytes"]
        assert record["returncode"] == 0
        assert record["rss_monitoring_error"] is None
        assert record["rss_exit_race_recovered"] is True
        assert record["stdout"]["path"]
        assert not base.psutil.pid_exists(record["pid"])
    finally:
        for process in launched:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


@pytest.mark.parametrize("error_type", [base.psutil.NoSuchProcess, base.psutil.AccessDenied])
@pytest.mark.parametrize("pid_kind", ["root", "unknown"])
def test_enumeration_fault_live_or_unknown_pid_is_fail_closed(tmp_path, monkeypatch, error_type, pid_kind):
    import json
    h = tooling()
    def children(process, **kwargs):
        raise error_type(process.pid if pid_kind == "root" else 2147483000, msg="enumeration fault")
    monkeypatch.setattr(base.psutil.Process, "children", children)
    monkeypatch.setattr(h, "final_peak_working_set", lambda _: pytest.fail("no final proof for monitoring faults"))
    with pytest.raises(base.AcceptanceError, match="RSS monitoring failed"):
        h.run_one([sys._base_executable, "-c", "import time; time.sleep(30)"], tmp_path, ROOT)
    record = json.loads((tmp_path / "process.json").read_text())
    assert record["rss_monitoring_error"]["phase"] == "descendants"
    assert record["rss_monitoring_error"]["type"] == error_type.__name__
    assert record["returncode"] != 0
    assert not base.psutil.pid_exists(record["pid"])


@pytest.mark.parametrize("peak", [0, -1, True, None])
def test_invalid_final_peak_fails_closed(tmp_path, monkeypatch, peak):
    h = tooling()
    original = subprocess.Popen
    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        assert process.wait(timeout=10) == 0
        return process
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(h, "final_peak_working_set", lambda _: peak)
    with pytest.raises(base.AcceptanceError, match="RSS monitoring failed"):
        h.run_one([sys._base_executable, "-c", "print('{}')"], tmp_path, ROOT)


@pytest.mark.parametrize("sampled,final,accepted", [(0,100,True), (100,100,True),
    (101,100,False), (100,101,False), (0,0,False), (0,None,False)])
def test_compact_separate_sample_and_final_peak_cap(sampled, final, accepted, monkeypatch):
    h = tooling()
    raw = synthetic_raw()
    raw.update(sampled_peak_rss_bytes=sampled, final_peak_working_set_bytes=final)
    monkeypatch.setattr(h, "MAX_RSS_BYTES", 100)
    if accepted:
        result = h.compact_run(raw, "moe-pool", "none", 100)
        assert result["sampled_peak_rss_bytes"] == sampled
        assert result["final_peak_working_set_bytes"] == final
    else:
        with pytest.raises(base.AcceptanceError):
            h.compact_run(raw, "moe-pool", "none", 100)

def test_main_orchestrates_synthetic_twelve_runs_without_native_launch(tmp_path, monkeypatch):
    """Synthetic control-flow test, NOT real acceptance evidence."""
    from test_expert_cache_acceptance import synthetic_metadata
    import json
    h = tooling()
    monkeypatch.setattr(h, "__file__", str(tmp_path / "scripts/moe_pool_acceptance.py"))
    inputs = tmp_path / "synthetic-input"
    inputs.write_bytes(b"synthetic")
    output = tmp_path / "build/issue87-unit-fixtures"
    monkeypatch.setattr(h.base, "parse_qxf", lambda _: synthetic_metadata())
    monkeypatch.setattr(h.base.psutil, "virtual_memory", lambda: type("Memory", (), {"available": 16 * 1024**3})())
    frozen_calls = []
    monkeypatch.setattr(h, "freeze", lambda *args: frozen_calls.append(1) or {"synthetic": True})
    calls = []
    monkeypatch.setattr(h, "run_one", lambda cmd, run_dir, cwd: calls.append(cmd) or {"synthetic": True})
    monkeypatch.setattr(h, "compact_run", lambda *args: {
        "wall_seconds": 1.0, "sampled_peak_rss_bytes": 100, "synthetic": True})
    monkeypatch.setattr(h, "equivalence", lambda _: {"synthetic": True})
    assert h.main(["--driver-exe", str(inputs), "--model", str(inputs), "--tokenizer", str(inputs),
                   "--output-dir", str(output)]) == 0
    assert len(calls) == 12
    assert len(frozen_calls) == 2
    report = json.loads((output / "report.json").read_text())
    assert report["performance_pass"] is False
    contract = report["fixed_contract"]
    assert contract["rss_scope"] == "root_process_working_set"
    assert contract["hard_os_memory_limit"] is False
    assert contract["lifetime_tree_guarantee"] is False
    assert contract["child_process_contract"] == "forbidden_if_observed"
    assert contract["rss_enforcement"] == "sampled_kill_and_post_exit_peak_rejection"
    assert all(len(c["warmups"]) == 1 and len(c["measured"]) == 2 for c in report["cells"])
    assert (output / "prompt.txt").read_text() == "Hi"
    assert report["qxf_budget_derivation"]["budget_bytes"] == base.derive_budget(synthetic_metadata(), 16 * 1024**3)["budget_bytes"]
    assert h.main(["--driver-exe", str(inputs), "--model", str(inputs), "--tokenizer", str(inputs),
                   "--output-dir", str(output)]) == 2
    assert len(calls) == 12


@pytest.mark.parametrize("changed", ["tokens", "steps", "rss", "returncode", "timeout"])
def test_compact_rejects_noncanonical_or_unbounded_run(changed):
    h = tooling()
    raw = synthetic_raw()
    if changed == "tokens":
        raw["payload"]["tokens"][0]["selected_token"] = 1
    elif changed == "steps":
        raw["payload"]["steps"] = 3
    elif changed == "rss":
        raw["sampled_peak_rss_bytes"] = 2 * 1024**3 + 1
    elif changed == "returncode":
        raw["returncode"] = 1
    else:
        raw["timed_out"] = True
    with pytest.raises(base.AcceptanceError):
        h.compact_run(raw, "moe-pool", "none", 100)


def invoke(driver, extra):
    exe, out = driver
    return subprocess.run([str(exe), "--model", "missing.qxf", "--tokenizer", "missing.qxt",
                           "--prompt-file", "missing-prompt", "--activation", "f32",
                           "--expert-cache-policy", "none", "--expert-cache-budget-bytes", "0",
                           "--dump-dir", str(out), "--ctx", "16", "--generate", "2",
                           "--io-backend", "buffered", *extra], cwd=out,
                          capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize("extra", [[], ["--thread-policy", "serial", "--threads", "1"],
                                      ["--thread-policy", "moe-pool", "--threads", "2"],
                                      ["--thread-policy", "moe-pool", "--threads", "64"]])
def test_driver_valid_policies_reach_prompt_io(driver, extra):
    result = invoke(driver, extra)
    assert result.returncode == 1
    assert "could not open prompt file" in result.stderr


@pytest.mark.parametrize("extra, message", [
    (["--thread-policy", "serial", "--threads", "2"], "one thread"),
    (["--thread-policy", "moe-pool", "--threads", "1"], "2..64"),
    (["--thread-policy", "moe-pool", "--threads", "65"], "2..64"),
    (["--thread-policy", "pool"], "unsupported thread policy"),
    (["--threads", "-1"], "invalid threads"),
])
def test_driver_rejects_invalid_policy_before_io(driver, extra, message):
    result = invoke(driver, extra)
    assert result.returncode == 2
    assert message in result.stderr
    assert "could not open" not in result.stderr
