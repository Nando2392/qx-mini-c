from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER_SOURCE = ROOT / "tests" / "native_expert_cache_v3_contract.c"
VCVARS = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)


def _msvc_command(command_file: Path, command: str) -> subprocess.CompletedProcess[str]:
    if sys.platform != "win32":
        pytest.skip("native expert-cache ABI compile tests require Windows")
    if not shutil.which("cmd.exe"):
        pytest.skip("cmd.exe is required for native expert-cache ABI compile tests")
    lines = ["@echo off", "setlocal"]
    if not shutil.which("cl"):
        if not VCVARS.is_file():
            pytest.skip("MSVC C17 compiler is required for the supported Windows ABI contract")
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


def _assert_compiled(completed: subprocess.CompletedProcess[str]) -> None:
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_public_header_preserves_legacy_layout_and_appends_v3(tmp_path: Path) -> None:
    executable = tmp_path / "native_expert_cache_v3_layout.exe"
    object_dir = f"{tmp_path.as_posix()}/"
    completed = _msvc_command(
        tmp_path / "compile_layout.cmd",
        "cl /nologo /std:c17 /W4 /WX /D_CRT_SECURE_NO_WARNINGS "
        "/DQX_EXPERT_CACHE_V3_HEADER_LAYOUT_ONLY /Iinclude "
        f'"{DRIVER_SOURCE}" /Fo:"{object_dir}" /Fe:"{executable.as_posix()}" /link /Brepro',
    )
    _assert_compiled(completed)
    run = subprocess.run([str(executable)], cwd=ROOT, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr


def test_native_expert_cache_v3_runtime_contract(tmp_path: Path) -> None:
    avx_object = tmp_path / "qx_avx2.obj"
    executable = tmp_path / "native_expert_cache_v3_contract.exe"
    object_dir = f"{tmp_path.as_posix()}/"
    command = (
        "cl /nologo /std:c17 /O2 /W4 /WX /arch:AVX2 /D_CRT_SECURE_NO_WARNINGS "
        f'/Iinclude /c src\\qx_avx2.c /Fo:"{avx_object.as_posix()}" && '
        "cl /nologo /std:c17 /O2 /W4 /WX /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc "
        "src\\qx_format.c src\\qx_expert_cache.c src\\qx_gguf.c src\\qx_tokenizer.c "
        "src\\qx_cuda_final_head_stub.c "
        f'"{DRIVER_SOURCE}" "{avx_object.as_posix()}" /Fo:"{object_dir}" '
        f'/Fe:"{executable.as_posix()}" /link /Brepro'
    )
    compiled = _msvc_command(tmp_path / "compile_contract.cmd", command)
    _assert_compiled(compiled)

    run = subprocess.run(
        [str(executable)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stderr == ""
    assert run.stdout == "native expert cache v3 ABI contract: pass\n"
