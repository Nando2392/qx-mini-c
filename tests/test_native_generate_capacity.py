from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QXQXF = ROOT / "build" / "qxqxf.exe"
MODEL = ROOT / "models" / "Qwen3-30B-A3B-UD-IQ2_M.qxf"
TOKENIZER = ROOT / "models" / "Qwen3-30B-A3B.qxt"


def require_or_skip(paths: tuple[Path, ...], purpose: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return
    message = f"{purpose} requires local assets: " + ", ".join(str(path) for path in missing)
    if os.environ.get("QX_REQUIRE_NATIVE_GENERATE") == "1":
        pytest.fail(message)
    pytest.skip(message)


def command(model: Path, tokenizer: Path, prompt: Path, *, max_tokens: int, ctx: int, extra: tuple[str, ...] = ()) -> list[str]:
    return [
        str(QXQXF),
        "generate",
        "--in",
        str(model),
        "--tokenizer",
        str(tokenizer),
        "--text-file",
        str(prompt),
        "--max-tokens",
        str(max_tokens),
        "--ctx",
        str(ctx),
        *extra,
    ]


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )


@pytest.fixture(scope="module", autouse=True)
def native_executable() -> None:
    require_or_skip((QXQXF,), "native generate capacity CLI tests")


def test_generate_accepts_max_tokens_above_legacy_64_before_file_io(tmp_path: Path) -> None:
    completed = run(
        command(
            tmp_path / "missing-model.qxf",
            tmp_path / "missing-tokenizer.qxt",
            tmp_path / "missing-prompt.txt",
            max_tokens=65,
            ctx=65,
        )
    )

    assert completed.returncode == 1
    assert "invalid --max-tokens" not in completed.stderr
    control = run(command(
        tmp_path / "missing-model.qxf", tmp_path / "missing-tokenizer.qxt",
        tmp_path / "missing-prompt.txt", max_tokens=1, ctx=1,
    ))
    assert control.returncode == 1
    assert completed.stderr == control.stderr
    assert completed.stderr.startswith("generate failed: ")


def test_generate_rejects_capacity_overflow_before_model_io(tmp_path: Path) -> None:
    require_or_skip((TOKENIZER,), "native generate capacity preflight test")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")
    missing_model = tmp_path / "missing-model.qxf"

    exact = run(command(missing_model, TOKENIZER, prompt, max_tokens=4095, ctx=4096))
    overflow = run(command(missing_model, TOKENIZER, prompt, max_tokens=4096, ctx=4096))

    assert exact.returncode == 1
    assert "prompt and generation exceed --ctx" not in exact.stderr
    assert "failed to open" in exact.stderr or "No such file or directory" in exact.stderr
    assert overflow.returncode == 2
    assert overflow.stderr == "generate failed: prompt and generation exceed --ctx\n"


def test_capacity_profile_is_separate_opt_in_with_actual_forward_counters(tmp_path: Path) -> None:
    require_or_skip((MODEL, TOKENIZER), "native generate capacity-profile acceptance test")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Hello!", encoding="utf-8", newline="")

    completed = run(command(MODEL, TOKENIZER, prompt, max_tokens=1, ctx=2, extra=("--capacity-profile",)))

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert set(payload) == {
        "prompt_token_count",
        "generated_token_ids",
        "generated_text",
        "stop_reason",
        "timing",
        "capacity_profile",
    }
    assert payload["capacity_profile"] == {
        "executed_forward_steps": 2,
        "prompt_forward_steps": 2,
        "generated_input_forward_steps": 0,
        "first_eos_output_index": None,
    }
