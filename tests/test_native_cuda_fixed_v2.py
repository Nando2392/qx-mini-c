from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER_SOURCE = ROOT / "tests" / "native_cuda_fixed_v2_contract.c"
VCVARS = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")


def _msvc_command(command_file: Path, command: str) -> subprocess.CompletedProcess[str]:
    lines = ["@echo off", "setlocal"]
    if not shutil.which("cl"):
        if not VCVARS.is_file():
            pytest.skip("MSVC C17 compiler is required for the supported Windows ABI contract")
        lines.extend((f'call "{VCVARS}"', "if errorlevel 1 exit /b %errorlevel%"))
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


def test_native_cuda_fixed_v2_contract(tmp_path: Path) -> None:
    avx_object = tmp_path / "qx_avx2.obj"
    executable = tmp_path / "native_cuda_fixed_v2_contract.exe"
    object_dir = f"{tmp_path.as_posix()}/"
    command = (
        "cl /nologo /std:c17 /O2 /W4 /WX /arch:AVX2 /D_CRT_SECURE_NO_WARNINGS "
        f'/Iinclude /c src\\qx_avx2.c /Fo:"{avx_object.as_posix()}" && '
        "cl /nologo /std:c17 /O2 /W4 /WX /D_CRT_SECURE_NO_WARNINGS /Iinclude "
        "src\\qx_format.c src\\qx_gguf.c src\\qx_tokenizer.c src\\qx_cuda_final_head_stub.c "
        f'"{DRIVER_SOURCE}" "{avx_object.as_posix()}" /Fo:"{object_dir}" '
        f'/Fe:"{executable.as_posix()}" /link /Brepro'
    )
    compiled = _msvc_command(tmp_path / "compile_fixed_v2.cmd", command)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    run = subprocess.run([str(executable)], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="strict", check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stderr == ""
    assert run.stdout == "native fixed CUDA v2 contract: pass\n"