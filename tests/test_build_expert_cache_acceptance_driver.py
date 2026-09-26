from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_expert_cache_acceptance_driver.cmd"
DRIVER = ROOT / "build" / "issue86-driver-testonly" / "expert_cache_acceptance_driver.exe"
VCVARS = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)


def _require_windows_msvc() -> None:
    if sys.platform != "win32":
        pytest.skip("expert-cache acceptance driver build requires Windows")
    if not shutil.which("cmd.exe") or not VCVARS.is_file():
        pytest.skip("MSVC Build Tools are required for the expert-cache acceptance driver build")


def test_build_driver_resolves_repo_from_foreign_working_directory(tmp_path: Path) -> None:
    _require_windows_msvc()
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(BUILD_SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert DRIVER.is_file()
    script = BUILD_SCRIPT.read_text(encoding="utf-8").lower()
    assert str(ROOT).replace("/", "\\").lower() not in script
    assert "%~dp0.." in script


def test_build_driver_skips_before_invoking_cmd_on_non_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def unexpected_run(*args: object, **kwargs: object) -> None:
        pytest.fail("cmd.exe must not be invoked on non-Windows platforms")

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    with pytest.raises(pytest.skip.Exception, match="Windows"):
        test_build_driver_resolves_repo_from_foreign_working_directory(tmp_path)
