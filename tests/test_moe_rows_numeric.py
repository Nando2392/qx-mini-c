"""Postimplementation numeric gate for packed F32 expert row scheduling (#87)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests" / "moe_rows_numeric_driver.c"
VCVARS = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)


@pytest.fixture(scope="module")
def numeric_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if sys.platform != "win32":
        pytest.skip("native expert row pool requires Windows")
    assert shutil.which("cmd.exe"), "cmd.exe is required"
    output = tmp_path_factory.mktemp("moe_rows_numeric_build")
    avx_object = output / "qx_avx2.obj"
    executable = output / "moe_rows_numeric_driver.exe"
    lines = ["@echo off", "setlocal"]
    if not shutil.which("cl"):
        assert VCVARS.is_file(), "MSVC C17 compiler is required"
        lines.extend((f'call "{VCVARS}" >nul', "if errorlevel 1 exit /b %errorlevel%"))
    lines.extend((
        "cl /nologo /std:c17 /O2 /W4 /WX /arch:AVX2 /D_CRT_SECURE_NO_WARNINGS "
        f'/Iinclude /c src\\qx_avx2.c /Fo:"{avx_object.as_posix()}"',
        "if errorlevel 1 exit /b %errorlevel%",
        "cl /nologo /std:c17 /O2 /W4 /WX /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc "
        "src\\qx_expert_cache.c src\\qx_gguf.c src\\qx_tokenizer.c src\\qx_cuda_final_head_stub.c "
        f'"{DRIVER}" "{avx_object.as_posix()}" /Fo:"{output.as_posix()}/" '
        f'/Fe:"{executable.as_posix()}" /link /Brepro',
        "exit /b %errorlevel%",
    ))
    command_file = output / "compile_numeric.cmd"
    command_file.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\r\n")
    compiled = subprocess.run(
        ["cmd.exe", "/d", "/c", str(command_file)], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False, timeout=120,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


@pytest.mark.parametrize("workers", [2, 3])
def test_packed_expert_rows_independent_numeric_gate(
    numeric_driver: Path, tmp_path: Path, workers: int,
) -> None:
    run = subprocess.run(
        [str(numeric_driver), str(workers)], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", errors="strict",
        check=False, timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stderr == ""
    assert run.stdout == (
        f"workers={workers} serial_calls=96 pool_jobs=96 "
        "numeric_failures=0 status_failures=0 setup_ok=1\n"
    )
