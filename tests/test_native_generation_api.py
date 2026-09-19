from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "build" / "native_generation_api_contract.exe"
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
def native_generation_driver() -> Path:
    require_or_skip((DRIVER,), "native generation API contract tests")
    return DRIVER


@pytest.fixture(scope="module")
def real_native_model(native_generation_driver: Path) -> Path:
    require_or_skip((MODEL,), "native generation API real-model tests")
    return MODEL


def run_driver(driver: Path, model: Path, eos_token_id: int, max_tokens: int = 2, ctx_tokens: int = 3) -> dict:
    completed = subprocess.run(
        [str(driver), str(model), str(eos_token_id), str(max_tokens), str(ctx_tokens)],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )
    assert completed.stdout.endswith("\n")
    assert completed.stdout.count("\n") == 1
    return json.loads(completed.stdout)


def assert_timing_contract(payload: dict) -> None:
    assert set(payload["timing"]) == {"prefill_seconds", "decode_seconds"}
    assert payload["timing"]["prefill_seconds"] >= 0
    assert payload["timing"]["decode_seconds"] > 0


def test_native_generation_api_stops_on_first_token_eos(
    native_generation_driver: Path, real_native_model: Path
) -> None:
    payload = run_driver(native_generation_driver, real_native_model, eos_token_id=358)

    assert set(payload) == {"token_ids", "token_count", "stopped_on_eos", "timing"}
    assert payload["token_ids"] == [358]
    assert payload["token_count"] == 1
    assert payload["stopped_on_eos"] is True
    assert_timing_contract(payload)


def test_native_generation_api_stops_on_second_token_eos(
    native_generation_driver: Path, real_native_model: Path
) -> None:
    payload = run_driver(native_generation_driver, real_native_model, eos_token_id=1184)

    assert payload["token_ids"] == [358, 1184]
    assert payload["token_count"] == 2
    assert payload["stopped_on_eos"] is True
    assert_timing_contract(payload)


def test_native_generation_api_without_eos_runs_to_max_tokens(
    native_generation_driver: Path, real_native_model: Path
) -> None:
    payload = run_driver(native_generation_driver, real_native_model, eos_token_id=-1)

    assert payload["token_ids"] == [358, 1184]
    assert payload["token_count"] == 2
    assert payload["stopped_on_eos"] is False
    assert_timing_contract(payload)


@pytest.mark.parametrize(
    ("max_tokens", "ctx_tokens", "expected_error"),
    [
        (0, 3, "invalid native generation argument"),
        (65, 66, "prompt and max_tokens in 1..64"),
        (2, 2, "prompt plus output must fit the 64-step and context limits"),
        (2, 0, "prompt plus output must fit the 64-step and context limits"),
    ],
)
def test_native_generation_api_rejects_invalid_limits_before_model_io(
    native_generation_driver: Path,
    tmp_path: Path,
    max_tokens: int,
    ctx_tokens: int,
    expected_error: str,
) -> None:
    missing_model = tmp_path / "missing-model.qxf"
    completed = subprocess.run(
        [str(native_generation_driver), str(missing_model), "-1", str(max_tokens), str(ctx_tokens)],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )

    assert completed.returncode == 1
    assert expected_error in completed.stderr
    assert "failed to open" not in completed.stderr
    assert "No such file or directory" not in completed.stderr
