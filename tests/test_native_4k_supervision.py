from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "native_4k_acceptance.py"
SPEC = importlib.util.spec_from_file_location("native_4k_supervision", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _tree_command(tmp_path: Path, mode: str, *, startup_delay: float = 0.0) -> tuple[list[str], Path, Path]:
    child_pid = tmp_path / "child.pid"
    grandchild_pid = tmp_path / "grandchild.pid"
    ready = tmp_path / "tree.ready"
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        "import os,sys,time\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.py"
    child.write_text(
        "import os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(float(sys.argv[5]))\n"
        "subprocess.Popen([sys.executable, sys.argv[2], sys.argv[3]])\n"
        "deadline=time.time()+2\n"
        "while not Path(sys.argv[3]).exists() and time.time()<deadline: time.sleep(.01)\n"
        "mode=sys.argv[4]\n"
        "if mode != 'parent-zero': os.write(1,b'stdout-exact-tail\\n'); os.write(2,b'stderr-exact-tail\\n')\n"
        "if mode != 'parent-zero': os.fsync(1); os.fsync(2)\n"
        "ready_tmp=Path(sys.argv[6]+'.tmp')\n"
        "ready_tmp.write_text('ready', encoding='utf-8')\n"
        "os.replace(ready_tmp, sys.argv[6])\n"
        "if mode == 'nonzero': raise SystemExit(7)\n"
        "if mode == 'parent-zero': os.write(1,b'{}'); raise SystemExit(0)\n"
        "if mode == 'rss': allocation=bytearray(96 << 20)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    return [sys.executable, str(child), str(child_pid), str(grandchild), str(grandchild_pid), mode,
            str(startup_delay), str(ready)], child_pid, grandchild_pid


def _read_pid(path: Path) -> int:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if path.exists():
            return int(path.read_text())
        time.sleep(0.01)
    raise AssertionError(f"PID file was not written: {path}")


def _assert_gone(*paths: Path) -> None:
    pids = [_read_pid(path) for path in paths]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and any(psutil.pid_exists(pid) for pid in pids):
        time.sleep(0.02)
    assert not [pid for pid in pids if psutil.pid_exists(pid)]


@contextmanager
def _cancel_after_ready(ready: Path, *, readiness_deadline_seconds: float = 4.0) -> Iterator[Callable[[], bool]]:
    cancel = threading.Event()
    stop_watcher = threading.Event()
    failures: list[AssertionError] = []

    def watch_readiness() -> None:
        deadline = time.monotonic() + readiness_deadline_seconds
        while not stop_watcher.is_set():
            if ready.exists():
                cancel.set()
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failures.append(AssertionError(f"process tree did not become ready: {ready}"))
                cancel.set()
                return
            stop_watcher.wait(min(0.01, remaining))

    watcher = threading.Thread(target=watch_readiness, name="cancel-after-tree-ready")
    watcher.start()
    try:
        yield cancel.is_set
    finally:
        stop_watcher.set()
        watcher.join(timeout=1.0)
        assert not watcher.is_alive(), "readiness watcher did not stop"
        if failures:
            raise failures[0]


@pytest.mark.parametrize("mode", ["timeout", "cancel", "rss", "nonzero", "exception"])
def test_failures_kill_child_and_grandchild_and_keep_exact_tails(tmp_path: Path, mode: str) -> None:
    command, child_pid, grandchild_pid = _tree_command(tmp_path, mode)
    kwargs = dict(deadline_seconds=3.0, rss_ceiling_bytes=1 << 30,
                  sample_interval_seconds=0.02, heartbeat_seconds=0.05)
    if mode == "timeout":
        kwargs["deadline_seconds"] = 0.25
    elif mode == "rss":
        # Let both interpreters publish PIDs/tails before deliberate allocation.
        kwargs["rss_ceiling_bytes"] = 96 << 20
    elif mode == "exception":
        def explode() -> bool:
            if grandchild_pid.exists():
                raise RuntimeError("supervisor callback exploded")
            return False
        kwargs["cancel_requested"] = explode

    expected = RuntimeError if mode == "exception" else runner.RunGateError
    if mode == "cancel":
        with _cancel_after_ready(tmp_path / "tree.ready") as cancel_requested:
            kwargs["cancel_requested"] = cancel_requested
            with pytest.raises(expected):
                runner.run_supervised(command, tmp_path, **kwargs)
    else:
        with pytest.raises(expected):
            runner.run_supervised(command, tmp_path, **kwargs)

    _assert_gone(child_pid, grandchild_pid)
    assert (tmp_path / "stdout.bin").read_bytes().endswith(b"stdout-exact-tail\n")
    assert (tmp_path / "stderr.bin").read_bytes().endswith(b"stderr-exact-tail\n")
    journal = json.loads((tmp_path / "phase-journal.json").read_text())
    assert journal["status"] not in {"completed", "pass"}


def test_cancel_waits_for_tree_readiness_before_killing_delayed_descendants(tmp_path: Path) -> None:
    command, child_pid, grandchild_pid = _tree_command(tmp_path, "cancel", startup_delay=1.0)
    ready = tmp_path / "tree.ready"
    with _cancel_after_ready(ready) as cancel_requested:
        with pytest.raises(runner.RunGateError):
            runner.run_supervised(
                command,
                tmp_path,
                deadline_seconds=6.0,
                rss_ceiling_bytes=1 << 30,
                sample_interval_seconds=0.02,
                heartbeat_seconds=0.05,
                cancel_requested=cancel_requested,
            )

    assert ready.read_text(encoding="utf-8") == "ready"
    _assert_gone(child_pid, grandchild_pid)
    assert (tmp_path / "stdout.bin").read_bytes().endswith(b"stdout-exact-tail\n")
    assert (tmp_path / "stderr.bin").read_bytes().endswith(b"stderr-exact-tail\n")
    assert json.loads((tmp_path / "phase-journal.json").read_text())["status"] == "cancelled"


def test_parent_exit_zero_with_live_descendant_is_nonpass_and_descendant_is_killed(tmp_path: Path) -> None:
    command, child_pid, grandchild_pid = _tree_command(tmp_path, "parent-zero")
    with pytest.raises(runner.RunGateError, match="descendant"):
        runner.run_supervised(command, tmp_path, deadline_seconds=2, rss_ceiling_bytes=1 << 30,
                              sample_interval_seconds=0.02, heartbeat_seconds=0.05)
    _assert_gone(child_pid, grandchild_pid)
    assert json.loads((tmp_path / "phase-journal.json").read_text())["status"] != "completed"


def test_cleanup_failure_is_terminal_nonpass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command, _, _ = _tree_command(tmp_path, "timeout")
    real_stop = runner._stop_tree
    def fail_after_cleanup(process, *args, **kwargs):
        real_stop(process, *args, **kwargs)
        raise runner.RunGateError("process-tree cleanup failed")
    monkeypatch.setattr(runner, "_stop_tree", fail_after_cleanup)
    with pytest.raises(runner.RunGateError, match="cleanup failed"):
        runner.run_supervised(command, tmp_path, deadline_seconds=0.2,
                              sample_interval_seconds=0.02, heartbeat_seconds=0.05)
    assert json.loads((tmp_path / "phase-journal.json").read_text())["status"] == "cleanup_failed"


def test_completed_journal_write_failure_never_returns_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    child = tmp_path / "ok.py"
    child.write_text("print('{}')\n", encoding="utf-8")
    real_journal = runner._journal
    def fail_completed(output_dir, phase, status, started, peak, **extra):
        if status == "completed":
            raise OSError("journal storage failed")
        return real_journal(output_dir, phase, status, started, peak, **extra)
    monkeypatch.setattr(runner, "_journal", fail_completed)
    with pytest.raises(OSError, match="journal storage"):
        runner.run_supervised([sys.executable, str(child)], tmp_path, deadline_seconds=2,
                              sample_interval_seconds=0.02, heartbeat_seconds=0.05)
    assert json.loads((tmp_path / "phase-journal.json").read_text())["status"] == "failed"


def test_main_prepare_exception_gets_terminal_failed_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "evidence"
    exe, model, tokenizer = (tmp_path / name for name in ("qx.exe", "model.qxf", "tokenizer.qxt"))
    for path in (exe, model, tokenizer):
        path.write_bytes(b"x")
    monkeypatch.setattr(runner, "construct_exact_prompt", lambda *a, **k: (_ for _ in ()).throw(ValueError("prepare boom")))
    monkeypatch.setattr(runner, "portable_path", lambda path, root: Path(path).name)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: type("D", (), {"free": 1 << 40})())
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output-dir", str(output), "--qx-exe", str(exe),
                                      "--model", str(model), "--tokenizer", str(tokenizer), "--prepare-only"])
    assert runner.main() == 2
    journal = json.loads((output / "phase-journal.json").read_text())
    assert (journal["phase"], journal["status"]) == ("prepare", "failed")
    assert not (output / "report.json").exists()


def test_main_preserves_terminal_native_timeout_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "evidence"
    exe, model, tokenizer = (tmp_path / name for name in ("qx.exe", "model.qxf", "tokenizer.qxt"))
    for path in (exe, model, tokenizer):
        path.write_bytes(b"x")
    monkeypatch.setattr(runner, "portable_path", lambda path, root: Path(path).name)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: type("D", (), {"free": 1 << 40})())
    monkeypatch.setattr(runner, "construct_exact_prompt", lambda *a, **k: {"text": "a", "token_count": 4095, "token_ids": [1], "attempts": 1})
    monkeypatch.setattr(runner, "tokenizer_encode", lambda *a, **k: {"token_count": 1, "token_ids": [1]})
    monkeypatch.setattr(runner, "freeze_inputs", lambda *a, **k: {"snapshot": 1})
    terminal = {"schema": runner.SCHEMA, "phase": "native", "status": "timeout",
                "elapsed_seconds": 28800.25, "peak_rss_bytes": 149401600,
                "heartbeat_interval_seconds": runner.HEARTBEAT_SECONDS,
                "computation_resume_supported": False, "returncode": 1}
    def timeout(*args, **kwargs):
        runner.atomic_json(output / "phase-journal.json", terminal)
        raise runner.RunGateError("native deadline exceeded")
    monkeypatch.setattr(runner, "run_supervised", timeout)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output-dir", str(output), "--qx-exe", str(exe),
                                      "--model", str(model), "--tokenizer", str(tokenizer)])
    assert runner.main() == 2
    assert json.loads((output / "phase-journal.json").read_text()) == terminal


def test_main_postrun_failure_gets_acceptance_failed_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "evidence"
    exe, model, tokenizer = (tmp_path / name for name in ("qx.exe", "model.qxf", "tokenizer.qxt"))
    for path in (exe, model, tokenizer):
        path.write_bytes(b"x")
    monkeypatch.setattr(runner, "portable_path", lambda path, root: Path(path).name)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: type("D", (), {"free": 1 << 40})())
    monkeypatch.setattr(runner, "construct_exact_prompt", lambda *a, **k: {"text": "a", "token_count": 4095, "token_ids": [1], "attempts": 1})
    monkeypatch.setattr(runner, "tokenizer_encode", lambda *a, **k: {"token_count": 1, "token_ids": [1]})
    snapshots = iter(({"snapshot": 1}, ValueError("post-freeze boom")))
    def freeze(*args, **kwargs):
        value = next(snapshots)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(runner, "freeze_inputs", freeze)
    monkeypatch.setattr(runner, "run_supervised", lambda *a, **k: {"payload": {}, "sampled_peak_rss_bytes": 123456789})
    monkeypatch.setattr(runner, "validate_native_run", lambda raw: {"case": True})
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output-dir", str(output), "--qx-exe", str(exe),
                                      "--model", str(model), "--tokenizer", str(tokenizer)])
    assert runner.main() == 2
    journal = json.loads((output / "phase-journal.json").read_text())
    assert (journal["phase"], journal["status"]) == ("acceptance", "failed")
    assert journal["peak_rss_bytes"] == 123456789
    assert not (output / "report.json").exists()
