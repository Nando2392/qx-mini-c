"""Opt-in real-model profile acceptance; never rebuild during frozen measurements."""
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import moe_pool_acceptance as acceptance


@pytest.mark.skipif(os.environ.get("QX_REQUIRE_MOE_POOL_REAL") != "1",
                    reason="real MoE pool profile gate is opt-in; build acceptance driver first")
@pytest.mark.parametrize("policy", ["serial", "moe-pool"])
def test_real_pool_profile_and_unchanged_serial_payload(tmp_path, policy):
    driver = ROOT / "build/issue86-driver-testonly/expert_cache_acceptance_driver.exe"
    model = ROOT / "models/Qwen3-30B-A3B-UD-IQ2_M.qxf"
    tokenizer = ROOT / "models/Qwen3-30B-A3B.qxt"
    assert all(p.is_file() for p in (driver, model, tokenizer))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(acceptance.base.PROMPT_TEXT, encoding="utf-8")
    command = acceptance.command_for(driver, model, tokenizer, prompt, tmp_path,
                                     policy, "none", 0)
    raw = acceptance.run_one(command, tmp_path, ROOT)
    if policy == "serial":
        assert "moe_parallel_jobs" not in raw["payload"]["thread_profile"]
        assert "moe_pool_profile" not in raw["payload"]
    result = acceptance.compact_run(raw, policy, "none", 0, tmp_path, tmp_path)
    assert result["generated_token_ids"] == acceptance.base.EXPECTED_F32_TOKENS
    assert result["moe_pool_profile"]["workers"] == (2 if policy == "moe-pool" else 0)
