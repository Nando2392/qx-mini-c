from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "build_cuda_msvc.bat"
RUNNER = ROOT / "tests" / "run_cuda_final_head_faults.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_cuda_final_head_faults", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_cuda_root(path: Path) -> Path:
    for relative in ("bin/nvcc.exe", "lib/x64/cudart.lib", "include/cuda_runtime.h"):
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
    return path


def test_build_script_rejects_missing_cuda_root_before_build(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "build_cuda_msvc.bat").write_bytes(BUILD_SCRIPT.read_bytes())
    (sandbox / "build_msvc.bat").write_text(
        "@echo off\r\necho UNEXPECTED_BUILD\r\nexit /b 47\r\n", encoding="ascii", newline=""
    )
    env = os.environ.copy()
    env.pop("QX_CUDA_ROOT", None)
    env.pop("CUDA_PATH", None)

    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "build_cuda_msvc.bat", "cuda"],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 3
    assert "QX_CUDA_ROOT" in completed.stdout
    assert "CUDA_PATH" in completed.stdout
    assert "UNEXPECTED_BUILD" not in completed.stdout


def test_build_script_accepts_explicit_valid_cuda_root_before_build(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "build_cuda_msvc.bat").write_bytes(BUILD_SCRIPT.read_bytes())
    (sandbox / "build_msvc.bat").write_text(
        "@echo off\r\necho EXPLICIT_ROOT_ACCEPTED\r\nexit /b 47\r\n", encoding="ascii", newline=""
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "cl.cmd").write_text("@exit /b 0\r\n", encoding="ascii", newline="")
    cuda_root = _fake_cuda_root(tmp_path / "cuda")
    env = os.environ.copy()
    env["QX_CUDA_ROOT"] = str(cuda_root)
    env.pop("CUDA_PATH", None)
    env["PATH"] = str(tools) + os.pathsep + env.get("PATH", "")

    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "build_cuda_msvc.bat", "cuda"],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 47
    assert "EXPLICIT_ROOT_ACCEPTED" in completed.stdout


def test_runner_requires_explicit_cuda_root(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    monkeypatch.delenv("QX_CUDA_ROOT", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    with pytest.raises(runner.GateError, match=r"--cuda-root.*QX_CUDA_ROOT.*CUDA_PATH"):
        runner.resolve_cuda_root(None)


def test_runner_prefers_cli_and_validates_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    cli_root = _fake_cuda_root(tmp_path / "cli-cuda")
    env_root = _fake_cuda_root(tmp_path / "env-cuda")
    monkeypatch.setenv("QX_CUDA_ROOT", str(env_root))
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "other-cuda"))

    assert runner.resolve_cuda_root(cli_root) == cli_root
