from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "build" / "native_generation_capacity_contract.exe"
MODEL = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"


def require_or_skip(paths: tuple[Path, ...], purpose: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return
    message = f"{purpose} requires local assets: " + ", ".join(str(path) for path in missing)
    if os.environ.get("QX_REQUIRE_NATIVE_GENERATE") == "1":
        pytest.fail(message)
    pytest.skip(message)


def run_driver(driver: Path, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(driver), *(str(arg) for arg in args)],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )


@pytest.fixture(scope="module")
def capacity_driver() -> Path:
    require_or_skip((DRIVER,), "native generation capacity API contract tests")
    return DRIVER


@pytest.fixture(scope="module")
def real_native_model(capacity_driver: Path) -> Path:
    require_or_skip((MODEL,), "native generation capacity API real-model tests")
    return MODEL


def test_native_generation_capacity_preflight_contract(capacity_driver: Path) -> None:
    completed = run_driver(capacity_driver)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "native generation capacity contract: pass\n"


@pytest.mark.parametrize(
    ("eos_token_id", "token_ids", "stopped_on_eos", "first_eos_output_index", "executed", "generated"),
    [
        (358, [358], True, 0, 2, 0),
        (1184, [358, 1184], True, 1, 3, 1),
        (-1, [358, 1184], False, None, 3, 1),
    ],
)
def test_native_generation_capacity_real_eos_and_forward_counters(
    capacity_driver: Path,
    real_native_model: Path,
    eos_token_id: int,
    token_ids: list[int],
    stopped_on_eos: bool,
    first_eos_output_index: int | None,
    executed: int,
    generated: int,
) -> None:
    completed = run_driver(capacity_driver, real_native_model, eos_token_id)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.endswith("\n")
    assert completed.stdout.count("\n") == 1
    payload = json.loads(completed.stdout)
    assert set(payload) == {
        "token_ids",
        "token_count",
        "stopped_on_eos",
        "first_eos_output_index",
        "executed_forward_steps",
        "prompt_forward_steps",
        "generated_input_forward_steps",
        "sampled_steps",
    }
    assert payload["token_ids"] == token_ids
    assert payload["token_count"] == len(token_ids)
    assert payload["stopped_on_eos"] is stopped_on_eos
    assert payload["first_eos_output_index"] == first_eos_output_index
    assert payload["executed_forward_steps"] == executed
    assert payload["prompt_forward_steps"] == 2
    assert payload["generated_input_forward_steps"] == generated
    assert payload["sampled_steps"] == len(token_ids)
