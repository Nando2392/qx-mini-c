import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "qxqxf.exe"
QXT = ROOT / "models" / "Qwen3-30B-A3B.qxt"
MATRIX_PATH = ROOT / "scripts" / "run_unicode_chat_greedy_matrix.py"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "qwen3-unicode-chat-greedy-matrix.json"


def load_matrix_module():
    spec = importlib.util.spec_from_file_location("unicode_chat_greedy_matrix", MATRIX_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(MATRIX_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_qwen3_chat_template_renders_exact_bytes(tmp_path):
    system = tmp_path / "system.txt"
    user = tmp_path / "user.txt"
    system.write_text("Responde en español.", encoding="utf-8", newline="")
    user.write_text("¿Qué significa café ☕?", encoding="utf-8", newline="")

    rendered = json.loads(subprocess.check_output([
        str(EXE),
        "chat-template-render",
        "--message", f"system:{system}",
        "--message", f"user:{user}",
        "--add-generation-prompt",
    ], text=True, encoding="utf-8"))

    expected = (
        "<|im_start|>system\nResponde en español.<|im_end|>\n"
        "<|im_start|>user\n¿Qué significa café ☕?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert rendered == {
        "template": "qwen3-chatml-subset-v1",
        "message_count": 2,
        "add_generation_prompt": True,
        "utf8_bytes": len(expected.encode("utf-8")),
        "text": expected,
    }


def test_qwen3_chat_template_rejects_invalid_role_and_utf8(tmp_path):
    text = tmp_path / "message.txt"
    text.write_text("hello", encoding="utf-8")
    role = subprocess.run(
        [str(EXE), "chat-template-render", "--message", f"tool:{text}"],
        text=True,
        capture_output=True,
    )
    assert role.returncode != 0
    assert "unsupported chat role" in role.stderr

    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"ok\xff")
    utf8 = subprocess.run(
        [str(EXE), "chat-template-render", "--message", f"user:{invalid}"],
        text=True,
        capture_output=True,
    )
    assert utf8.returncode != 0
    assert "valid UTF-8" in utf8.stderr


def test_matrix_fixture_is_strict_and_separates_claims():
    matrix = load_matrix_module()
    fixture = matrix.load_contract(FIXTURE_PATH)
    assert len(fixture["tokenizer_cases"]) >= 7
    assert len(fixture["chat_cases"]) >= 2
    assert len(fixture["greedy_cases"]) >= 3
    assert fixture["claims"] == {
        "tokenizer_parity": "case_local",
        "chat_template_render": "case_local",
        "greedy_sequence": "case_and_modality_local",
        "logit_parity": "not_claimed",
        "semantic_equivalence": "not_claimed",
        "global_modality_equivalence": "not_claimed",
    }
    assert all(case["oracle_origin"] == "llama-tokenize" for case in fixture["tokenizer_cases"])
    assert all(case["oracle_origin"] == "llama_chat_apply_template" for case in fixture["chat_cases"])
    assert all(case["oracle_origin"] == "llama_sequence_oracle" for case in fixture["greedy_cases"])
    assert {"ascii-control", "latin1", "combining", "cjk", "arabic", "emoji"} <= {
        case["name"] for case in fixture["tokenizer_cases"]
    }
    assert all(isinstance(case["expected_token_ids"], list) for case in fixture["tokenizer_cases"])
    assert all(case["expected_decode_text"] == case["text"] for case in fixture["tokenizer_cases"])
    assert all(isinstance(case["expected_text"], str) for case in fixture["chat_cases"])
    assert all(isinstance(case["expected_token_ids"], list) for case in fixture["chat_cases"])
    assert fixture["chat_cases"][1]["messages"][-1]["role"] == "assistant"
    assert fixture["chat_cases"][1]["add_generation_prompt"] is False
    assert all(isinstance(case["expected_llama_f16_tokens"], list) for case in fixture["greedy_cases"])
    assert all(isinstance(case["expected_llama_q8_0_tokens"], list) for case in fixture["greedy_cases"])
    assert all(isinstance(case["expected_qx_tokens"], list) for case in fixture["greedy_cases"])
    assert fixture["source"]["model_size"] > 0
    assert fixture["source"]["qxf_size"] > 0


def test_matrix_runner_executes_exact_tokenizer_and_chat_claims(tmp_path):
    if not EXE.exists() or not QXT.exists():
        pytest.skip("real QX tokenizer fixtures are not available")
    matrix = load_matrix_module()
    fixture = matrix.load_contract(FIXTURE_PATH)
    report = matrix.run_contract(fixture, ROOT, tmp_path)
    assert report["schema"] == "qx-unicode-chat-greedy-report-v1"
    assert report["tokenizer"]["passed"] == len(fixture["tokenizer_cases"])
    assert report["chat_template"]["passed"] == len(fixture["chat_cases"])
    assert report["greedy"]["status"] in {"passed", "not_run"}
    assert report["overall_status"] in {"passed", "partial_environmental"}


def test_case_logit_metrics_keep_llama_modalities_separate(tmp_path):
    matrix = load_matrix_module()
    qx_dir = tmp_path / "qx"
    f16_dir = tmp_path / "llama-f16"
    q8_dir = tmp_path / "llama-q8"
    for directory in (qx_dir, f16_dir, q8_dir):
        directory.mkdir()
    values = {
        qx_dir: ([1.0, 3.0, 2.0], [3.0, 2.0, 1.0]),
        f16_dir: ([1.0, 3.0, 2.0], [3.0, 2.0, 1.0]),
        q8_dir: ([1.0, 2.0, 3.0], [2.0, 3.0, 1.0]),
    }
    for directory, steps in values.items():
        for step, logits in enumerate(steps):
            (directory / f"step-{step}-logits.f32").write_bytes(
                struct.pack(f"<{len(logits)}f", *logits)
            )

    metrics = matrix.compare_case_logits(
        qx_dir,
        {"f16": f16_dir, "q8_0": q8_dir},
        generation_steps=2,
    )

    assert [row["step"] for row in metrics] == [0, 1]
    assert all(row["llama_f16"]["argmax_match"] is True for row in metrics)
    assert all(row["llama_f16"]["pass"] is True for row in metrics)
    assert all(row["llama_q8_0"]["argmax_match"] is False for row in metrics)
    assert all(row["llama_q8_0"]["pass"] is False for row in metrics)


def test_activation_logit_metrics_keep_qx_modes_separate(tmp_path):
    matrix = load_matrix_module()
    f32_dir = tmp_path / "qx-f32"
    q8k_dir = tmp_path / "qx-q8k"
    llama_f16 = tmp_path / "llama-f16"
    llama_q8 = tmp_path / "llama-q8"
    for directory in (f32_dir, q8k_dir, llama_f16, llama_q8):
        directory.mkdir()
    values = {
        f32_dir: [1.0, 3.0, 2.0],
        q8k_dir: [1.0, 2.0, 3.0],
        llama_f16: [1.0, 3.0, 2.0],
        llama_q8: [1.0, 2.0, 3.0],
    }
    for directory, logits in values.items():
        (directory / "step-0-logits.f32").write_bytes(
            struct.pack(f"<{len(logits)}f", *logits)
        )

    report = matrix.compare_activation_logits(
        {"f32": f32_dir, "q8_k_compat": q8k_dir},
        {"f16": llama_f16, "q8_0": llama_q8},
        generation_steps=1,
    )

    assert tuple(report) == ("f32", "q8_k_compat")
    assert report["f32"][0]["llama_f16"]["pass"] is True
    assert report["f32"][0]["llama_q8_0"]["pass"] is False
    assert report["q8_k_compat"][0]["llama_f16"]["pass"] is False
    assert report["q8_k_compat"][0]["llama_q8_0"]["pass"] is True


def test_matrix_runner_reports_missing_artifacts_fail_closed(tmp_path):
    matrix = load_matrix_module()
    fixture = matrix.load_contract(FIXTURE_PATH)
    root = tmp_path / "root"
    (root / "build").mkdir(parents=True)
    (root / "build" / "qxqxf.exe").write_bytes(b"not executed")

    with pytest.raises(ValueError, match="Qwen3-30B-A3B.qxt"):
        matrix.run_contract(fixture, root, tmp_path / "work")


@pytest.mark.parametrize("mutation,match", [
    (lambda raw: raw.replace('"issue": 22', '"issue": true'), "issue"),
    (lambda raw: raw.replace('"schema":', '"extra": 1, "schema":', 1), "fields"),
    (lambda raw: raw.replace('"schema":', '"schema": "duplicate", "schema":', 1), "duplicate JSON key"),
    (lambda raw: raw.replace('"issue": 22', '"issue": NaN'), "non-finite JSON number"),
])
def test_matrix_loader_fails_closed_on_malformed_json(tmp_path, mutation, match):
    matrix = load_matrix_module()
    raw = FIXTURE_PATH.read_text(encoding="utf-8")
    malformed = tmp_path / "malformed.json"
    malformed.write_text(mutation(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        matrix.load_contract(malformed)


def test_matrix_runner_rejects_existing_output_and_external_paths(tmp_path):
    matrix = load_matrix_module()
    fixture = matrix.load_contract(FIXTURE_PATH)
    existing = tmp_path / "report.json"
    existing.write_text("trusted", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        matrix.publish_report(existing, {"ok": True})
    assert existing.read_text(encoding="utf-8") == "trusted"

    escaped = json.loads(json.dumps(fixture))
    escaped["source"]["model_path"] = "../outside.gguf"
    with pytest.raises(ValueError, match="model_path"):
        matrix.validate_contract(escaped)

    escaped_qxf = json.loads(json.dumps(fixture))
    escaped_qxf["source"]["qxf_path"] = "../outside.qxf"
    with pytest.raises(ValueError, match="qxf_path"):
        matrix.validate_contract(escaped_qxf)

    wrong_provenance = json.loads(json.dumps(fixture))
    wrong_provenance["source"]["llama_cpp_commit"] = "0" * 40
    with pytest.raises(ValueError, match="canonical tokenizer oracle"):
        matrix.validate_contract(wrong_provenance)

    wrong_model = json.loads(json.dumps(fixture))
    wrong_model["source"]["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="canonical tokenizer oracle"):
        matrix.validate_contract(wrong_model)

    wrong_revision = json.loads(json.dumps(fixture))
    wrong_revision["source"]["oracle_revision"] = "reviewed"
    with pytest.raises(ValueError, match="oracle_revision"):
        matrix.validate_contract(wrong_revision)

    wrong_args = json.loads(json.dumps(fixture))
    wrong_args["source"]["oracle_args"] = ["--ids"]
    with pytest.raises(ValueError, match="oracle_args"):
        matrix.validate_contract(wrong_args)


def test_chat_template_cli_fails_closed_on_role_utf8_and_message_limit(tmp_path):
    exe = ROOT / "build" / "qxqxf.exe"
    valid = tmp_path / "valid.txt"
    valid.write_text("hola", encoding="utf-8")
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xc3\x28")

    bad_role = subprocess.run(
        [str(exe), "chat-template-render", "--message", f"tool:{valid}"],
        text=True,
        capture_output=True,
    )
    assert bad_role.returncode != 0
    assert "unsupported chat role" in bad_role.stderr

    bad_utf8 = subprocess.run(
        [str(exe), "chat-template-render", "--message", f"user:{invalid}"],
        text=True,
        capture_output=True,
    )
    assert bad_utf8.returncode != 0
    assert "valid UTF-8" in bad_utf8.stderr

    too_many = [str(exe), "chat-template-render"]
    for _ in range(65):
        too_many.extend(["--message", f"user:{valid}"])
    overflow = subprocess.run(too_many, text=True, capture_output=True)
    assert overflow.returncode != 0
    assert "invalid --message" in overflow.stderr


def test_python_entrypoints_compile():
    subprocess.run([
        sys.executable,
        "-m",
        "py_compile",
        str(MATRIX_PATH),
    ], check=True)
