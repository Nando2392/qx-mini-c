from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from moe_pool_build import build_driver


@pytest.fixture(scope="module")
def native_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_driver(tmp_path_factory.mktemp("moe_pool_build"))


@pytest.mark.parametrize("api,name", tuple(enumerate(("fixed-v1", "fixed-v2", "fixed-v3", "buffer-v2", "buffer-v3"))))
def test_moe_pool_native_api_preflight(native_driver: Path, tmp_path: Path, api: int, name: str) -> None:
    run = subprocess.run(
        [str(native_driver), str(api)], cwd=tmp_path, capture_output=True, text=True,
        encoding="utf-8", errors="strict", check=False,
    )
    (tmp_path / "green.log").write_text(
        run.stdout + run.stderr + f"\nexit={run.returncode}\n", encoding="utf-8",
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stderr == ""
    assert run.stdout.endswith(f"{name}: MoE pool API preflight contract pass; no model executed\n")
    assert "policy accepted; missing input, no model execution" in run.stdout
    assert "policy rejected before input I/O" in run.stdout
