from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SUBPROCESS_TESTS = (
    ROOT / "tests" / "test_native_generate.py",
    ROOT / "tests" / "test_native_generate_binding.py",
    ROOT / "tests" / "test_native_generation_api.py",
)


def test_native_output_capture_decodes_utf8_strictly() -> None:
    failures: list[str] = []
    for path in NATIVE_SUBPROCESS_TESTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "subprocess"
                and function.attr == "run"
            ):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            capture = keywords.get("capture_output")
            if not (isinstance(capture, ast.Constant) and capture.value is True):
                continue
            encoding = keywords.get("encoding")
            errors = keywords.get("errors")
            if not (isinstance(encoding, ast.Constant) and encoding.value == "utf-8"):
                failures.append(f"{path.name}:{node.lineno} missing encoding='utf-8'")
            if not (isinstance(errors, ast.Constant) and errors.value == "strict"):
                failures.append(f"{path.name}:{node.lineno} missing errors='strict'")

    assert failures == []
