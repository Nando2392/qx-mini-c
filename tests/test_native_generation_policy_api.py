from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "build" / "native_generation_policy_api_contract.exe"
MODEL = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"


def require_or_skip(paths: tuple[Path, ...], purpose: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return
    message = f"{purpose} requires local assets: " + ", ".join(str(path) for path in missing)
    if os.environ.get("QX_REQUIRE_NATIVE_GENERATE") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def policy_driver() -> Path:
    require_or_skip((DRIVER,), "native generation policy API contract tests")
    return DRIVER


def run_contract(driver: Path, *args: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(driver), *(str(arg) for arg in args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_native_generation_policy_preflight_contract(policy_driver: Path) -> None:
    completed = run_contract(policy_driver)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "native generation policy API contract: pass\n"


def test_native_generation_policy_real_profile_and_global_restore(policy_driver: Path) -> None:
    require_or_skip((MODEL,), "native generation policy API real-model contract")
    completed = run_contract(policy_driver, MODEL)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "native generation policy real-model contract: pass\n"
