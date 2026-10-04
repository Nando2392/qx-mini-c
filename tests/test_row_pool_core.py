from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VCVARS = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")


def compile_driver(directory: Path, baseline: bool = False) -> Path:
    if sys.platform != "win32":
        pytest.skip("row pool contracts require Windows MSVC")
    if not shutil.which("cl") and not VCVARS.is_file():
        pytest.skip("MSVC C17 compiler is required")
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / ("serial_baseline.exe" if baseline else "row_pool_contract.exe")
    script = directory / ("red_compile.cmd" if baseline else "green_compile.cmd")
    lines = ["@echo off", "setlocal"]
    if not shutil.which("cl"):
        lines += [f'call "{VCVARS}" >nul', "if errorlevel 1 exit /b %errorlevel%"]
    define = "/DQX_ROW_POOL_SERIAL_BASELINE" if baseline else ""
    lines += [
        f'cl /nologo /std:c17 /W4 /WX /D_CRT_SECURE_NO_WARNINGS {define} /Isrc '
        f'tests\\row_pool_contract.c /Fo:"{directory.as_posix()}/" /Fe:"{exe.as_posix()}" /link /Brepro',
        "exit /b %errorlevel%",
    ]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\r\n")
    result = subprocess.run(["cmd.exe", "/d", "/c", str(script)], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=120)
    (directory / ("red_compile.log" if baseline else "green_compile.log")).write_text(
        result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    return exe


def run_contract(exe: Path, scenario: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([str(exe), scenario], cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", errors="strict", timeout=30)
    (exe.parent / f"{exe.stem}-{scenario}.log").write_text(
        f"returncode={result.returncode}\n" + result.stdout + result.stderr, encoding="utf-8")
    return result


@pytest.fixture(scope="module")
def driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return compile_driver(tmp_path_factory.mktemp("row-pool"))


@pytest.mark.parametrize("scenario", ["numeric", "boundaries", "drain", "faults", "partition"])
def test_row_pool_contract(driver: Path, scenario: str) -> None:
    result = run_contract(driver, scenario)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
    assert not result.stderr


def test_serial_baseline_rejected(tmp_path: Path) -> None:
    """Controlled semantic RED, not a missing-header/compiler failure."""
    result = run_contract(compile_driver(tmp_path, baseline=True), "numeric")
    assert result.returncode == 1
    assert "w.callbacks == 4" in result.stderr


@pytest.mark.parametrize("available", [True, False])
def test_header_without_fault_wrappers(tmp_path: Path, available: bool) -> None:
    if sys.platform != "win32":
        pytest.skip("header compile probes require Windows MSVC")
    if not shutil.which("cl") and not VCVARS.is_file():
        pytest.skip("MSVC C17 compiler is required")
    source = tmp_path / "probe.c"
    source.write_text('''#include "qx_row_pool.h"
#include "qx_row_pool.h"
static int visit(void *context, uint32_t begin, uint32_t end) {
    return context != NULL || begin >= end;
}
int main(void) {
    char err[2] = { 'x', 'x' };
    qx_row_pool *p = qx_row_pool_create(2, err, sizeof(err));
#ifdef _WIN32
    if (!p || qx_row_pool_workers(p) != 2 || qx_row_pool_jobs(p)) return 1;
    if (!qx_row_pool_run(p, 1, visit, NULL, err, sizeof(err))) return 2;
    if (qx_row_pool_jobs(p) != 1) return 3;
#else
    if (p || !err[0] || err[1]) return 4;
    if (qx_row_pool_run(p, 1, visit, NULL, err, sizeof(err))) return 5;
    if (qx_row_pool_workers(p) || qx_row_pool_jobs(p)) return 6;
#endif
    qx_row_pool_destroy(p);
    return 0;
}
''', encoding="utf-8")
    exe = tmp_path / "probe.exe"
    script = tmp_path / "probe.cmd"
    lines = ["@echo off", "setlocal"]
    if not shutil.which("cl"):
        lines += [f'call "{VCVARS}" >nul', "if errorlevel 1 exit /b %errorlevel%"]
    flags = "" if available else "/U_WIN32"
    lines += [f'cl /nologo /std:c17 /W4 /WX {flags} /Isrc "{source}" '
              f'/Fo:"{tmp_path.as_posix()}/" /Fe:"{exe}" /link /Brepro',
              "exit /b %errorlevel%"]
    script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\r\n")
    result = subprocess.run(["cmd.exe", "/d", "/c", str(script)], cwd=ROOT,
                            capture_output=True, text=True, timeout=120)
    (tmp_path / "probe-compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(exe)], cwd=ROOT, capture_output=True, text=True, timeout=30)
    (tmp_path / "probe-run.log").write_text(f"returncode={result.returncode}\n" + result.stdout + result.stderr,
                                             encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
