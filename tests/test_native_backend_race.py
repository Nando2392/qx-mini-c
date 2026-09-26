from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests" / "native_backend_race_driver.c"
VCVARS = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)


def _run_msvc(command_file: Path, command: str) -> subprocess.CompletedProcess[str]:
    if sys.platform != "win32":
        pytest.skip("native backend race regression requires Windows")
    assert shutil.which("cmd.exe"), "cmd.exe is required for the Windows native backend race regression"
    lines = ["@echo off", "setlocal"]
    if not shutil.which("cl"):
        assert VCVARS.is_file(), "MSVC C17 compiler is required for the supported Windows native API"
        lines.extend((f'call "{VCVARS}" >nul', "if errorlevel 1 exit /b %errorlevel%"))
    lines.extend((command, "exit /b %errorlevel%"))
    command_file.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\r\n")
    return subprocess.run(
        ["cmd.exe", "/d", "/c", str(command_file)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_native_options_backend_isolated_from_process_default(tmp_path: Path) -> None:
    avx_object = tmp_path / "qx_avx2.obj"
    executable = tmp_path / "native_backend_race_driver.exe"
    object_dir = f"{tmp_path.as_posix()}/"
    command = (
        "cl /nologo /std:c17 /O2 /W4 /WX /arch:AVX2 /D_CRT_SECURE_NO_WARNINGS "
        f'/Iinclude /c src\\qx_avx2.c /Fo:"{avx_object.as_posix()}" && '
        "cl /nologo /std:c17 /O2 /W4 /WX /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc "
        "src\\qx_expert_cache.c src\\qx_gguf.c src\\qx_tokenizer.c src\\qx_cuda_final_head_stub.c "
        f'"{DRIVER}" "{avx_object.as_posix()}" /Fo:"{object_dir}" '
        f'/Fe:"{executable.as_posix()}" /link /Brepro'
    )
    compiled = _run_msvc(tmp_path / "compile_native_backend_race.cmd", command)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    run = subprocess.run(
        [str(executable)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stderr == ""
    assert run.stdout == "native backend isolation: pass\n"
