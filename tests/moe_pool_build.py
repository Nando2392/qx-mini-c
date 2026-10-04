"""Isolated native MoE policy contract build; importing does not run a compiler."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER_SOURCE = ROOT / "tests" / "moe_pool_runtime_driver.c"


def _vcvars() -> Path | None:
    installer = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if installer.is_file():
        found = subprocess.run(
            [str(installer), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if found:
            candidate = Path(found) / "VC/Auxiliary/Build/vcvars64.bat"
            if candidate.is_file():
                return candidate
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        base = Path(os.environ.get(variable, "C:/Program Files (x86)")) / "Microsoft Visual Studio"
        for candidate in sorted(base.glob("*/*/VC/Auxiliary/Build/vcvars64.bat")):
            if candidate.is_file():
                return candidate
    return None


def build_driver(output_dir: Path, source: Path = DRIVER_SOURCE) -> Path:
    if sys.platform != "win32" or not shutil.which("cmd.exe"):
        pytest.skip("native MoE pool policy contract requires Windows and MSVC")
    lines = ["@echo off", "setlocal"]
    if not shutil.which("cl"):
        vcvars = _vcvars()
        if vcvars is None:
            pytest.skip("MSVC C17 compiler is required for the native MoE pool contract")
        lines.extend((f'call "{vcvars}" >nul', "if errorlevel 1 exit /b %errorlevel%"))
    output_dir.mkdir(parents=True, exist_ok=True)
    avx_object = output_dir / "qx_avx2.obj"
    executable = output_dir / "moe_pool_runtime_driver.exe"
    object_dir = output_dir.as_posix() + "/"
    lines.extend((
        "cl /nologo /std:c17 /O2 /W4 /WX /arch:AVX2 /D_CRT_SECURE_NO_WARNINGS "
        f'/Iinclude /c src\\qx_avx2.c /Fo:"{avx_object.as_posix()}"',
        "if errorlevel 1 exit /b %errorlevel%",
        "cl /nologo /std:c17 /O2 /W4 /WX /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc "
        "src\\qx_format.c src\\qx_expert_cache.c src\\qx_gguf.c src\\qx_tokenizer.c "
        "src\\qx_cuda_final_head_stub.c "
        f'"{source}" "{avx_object.as_posix()}" /Fo:"{object_dir}" '
        f'/Fe:"{executable.as_posix()}" /link /Brepro',
        "exit /b %errorlevel%",
    ))
    command_file = output_dir / "compile_contract.cmd"
    command_file.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\r\n")
    compiled = subprocess.run(
        ["cmd.exe", "/d", "/c", str(command_file)], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    (output_dir / "build.log").write_text(
        compiled.stdout + compiled.stderr + f"\nexit={compiled.returncode}\n", encoding="utf-8",
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable
