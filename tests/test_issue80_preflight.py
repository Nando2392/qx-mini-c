from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("field,minimum,maximum", [
    ("hidden", 1, 0xFFFFFFFF), ("vcur_count", 1, 0xFFFFFFFF),
    ("kqv_count", 1, 0xFFFFFFFF), ("layers", 4, 48),
    ("ctx", 2, 0xFFFFFFFF), ("seed", 0, 0xFFFFFFFF),
    ("continuation_token", 0, 0xFFFFFFFF),
])
def test_numeric_preflight_before_io(tmp_path, monkeypatch, field, minimum, maximum):
    script = Path(__file__).resolve().parents[1] / "scripts/run_issue80_layer3_seams.py"
    spec = importlib.util.spec_from_file_location("preflight_target", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        pytest.fail("I/O attempted before numeric validation")

    monkeypatch.setattr(module, "sha256", forbidden)
    monkeypatch.setattr(module, "_invoke", forbidden)
    for value in (minimum - 1, maximum + 1, True, 1.5, "4", None):
        args = Namespace(hidden=2048, vcur_count=512, kqv_count=4096,
                         layers=48, ctx=4, seed=7, continuation_token=1124,
                         out=tmp_path / "must-not-exist")
        setattr(args, field, value)
        with pytest.raises(ValueError, match=field):
            module.run(args)
        assert not args.out.exists()
