from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests" / "expert_cache_core_driver.c"
VCVARS = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)


def _run_msvc(command_file: Path, command: str) -> subprocess.CompletedProcess[str]:
    lines = ["@echo off", "setlocal"]
    if not shutil.which("cl"):
        lines.extend((f'call "{VCVARS}" >nul', "if errorlevel 1 exit /b %errorlevel%"))
    lines.extend((command, "exit /b %errorlevel%"))
    command_file.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\r\n")
    return subprocess.run(
        ["cmd.exe", "/d", "/c", str(command_file)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )


def _require_windows_msvc() -> None:
    if sys.platform != "win32":
        pytest.skip("expert-cache core compile tests require Windows")
    if not shutil.which("cmd.exe"):
        pytest.skip("cmd.exe is required for expert-cache core compile tests")
    if not shutil.which("cl") and not VCVARS.is_file():
        pytest.skip("MSVC C17 compiler is required for expert-cache core compile tests")


def _compile_expert_cache_driver(tmp_path: Path) -> Path:
    _require_windows_msvc()
    executable = tmp_path / "expert_cache_core_driver.exe"
    object_dir = f"{tmp_path.as_posix()}/"
    completed = _run_msvc(
        tmp_path / "compile.cmd",
        "cl /nologo /std:c17 /W4 /WX /D_CRT_SECURE_NO_WARNINGS /Isrc "
        "src\\qx_expert_cache.c tests\\expert_cache_core_driver.c "
        f'/Fo:"{object_dir}" /Fe:"{executable.as_posix()}" /link /Brepro',
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


@pytest.fixture(scope="module")
def expert_cache_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _compile_expert_cache_driver(tmp_path_factory.mktemp("expert-cache-core"))


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("lru", "requests=6 hits=2 misses=4 loads=4 evictions=2 resident=8 peak=8 read=16 avoided=8"),
        ("pinning", "requests=3 hits=0 misses=3 loads=2 evictions=0 resident=8 peak=8 read=8 avoided=0"),
        ("failures", "partial_read=2 truncated_read=3 allocation_reads=0 resident=0"),
        ("validation", "invalid_requests=0 oversize_requests=1 oversize_reads=0"),
        ("identity", "requests=2 hits=0 misses=2 loads=2 evictions=0 resident=8 peak=8 read=8 avoided=0"),
        ("isolation", "first_loads=1 second_loads=1 first_reads=4 second_reads=4"),
        ("destroy", "allocations=4 frees=4 resident_before_destroy=8"),
        ("lifetime", "frees=4 poison=1 stale_cleared=1"),
    ],
)
def test_expert_cache_core_contract(
    expert_cache_driver: Path, scenario: str, expected: str
) -> None:
    completed = subprocess.run(
        [str(expert_cache_driver), scenario],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == expected
    assert completed.stderr == ""


def test_core_driver_skips_before_invoking_cmd_on_non_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def unexpected_run(*args: object, **kwargs: object) -> None:
        pytest.fail("cmd.exe must not be invoked on non-Windows platforms")

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    with pytest.raises(pytest.skip.Exception, match="Windows"):
        _compile_expert_cache_driver(tmp_path)


def test_core_driver_propagates_supported_compiler_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("shutil.which", lambda executable: f"C:/tools/{executable}.exe")
    failed = subprocess.CompletedProcess(
        args=["cmd.exe"], returncode=2, stdout="compiler stdout\n", stderr="compiler failed\n"
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: failed)

    with pytest.raises(AssertionError, match="compiler failed"):
        _compile_expert_cache_driver(tmp_path)
