#!/usr/bin/env python3
"""Fail-closed loader and executable reporter for the Issue 22 matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from compare_logits import compare_logit_files

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ORACLE = ROOT / "tests" / "fixtures" / "qwen3-tokenizer-llama-cpp-goldens.json"
TOP_FIELDS = ("schema", "issue", "source", "tokenizer_cases", "chat_cases", "greedy_cases", "claims")
SOURCE_FIELDS = ("model_path", "model_sha256", "model_size", "qxf_path", "qxf_sha256", "qxf_size", "llama_cpp_commit", "oracle_revision", "oracle_args", "runtime")
RUNTIME_FIELDS = ("tokenizer", "chat_template", "llama_greedy", "qx_greedy")
CLAIM_FIELDS = ("tokenizer_parity", "chat_template_render", "greedy_sequence", "logit_parity", "semantic_equivalence", "global_modality_equivalence")
TOKENIZER_FIELDS = ("name", "text", "parse_special", "expected_token_ids", "expected_decode_text", "oracle_origin", "mode")
CHAT_FIELDS = ("name", "messages", "add_generation_prompt", "expected_text", "expected_token_ids", "oracle_origin", "mode")
MESSAGE_FIELDS = ("role", "content")
GREEDY_FIELDS = ("name", "prompt_token_ids", "generation_steps", "expected_llama_f16_tokens", "expected_llama_q8_0_tokens", "expected_qx_tokens", "oracle_origin", "mode")
TOKENIZER_ORDER = ("ascii-control", "latin1", "combining", "cjk", "arabic", "emoji", "mixed")
CHAT_ORDER = ("system-user-generation", "user-assistant-no-generation")
GREEDY_ORDER = ("token-42", "token-56", "token-1000")
ORACLE_ARGS = [
    {"component": "tokenizer", "args": ["-m", "<model>", "-p", "<text>", "--ids", "--no-bos", "--no-escape", "--no-parse-special"]},
    {"component": "chat_template", "args": ["llama_chat_apply_template", "source_gguf_template"]},
    {"component": "greedy", "args": ["prompt_token_ids=<case>", "steps=2", "temperature=0", "modes=f16,q8_0"]},
]
EXPECTED_RUNTIME = {
    "tokenizer": "cpu/vocab_only",
    "chat_template": "cpu/vocab_only",
    "llama_greedy": ["f16_cpu", "q8_0_cpu"],
    "qx_greedy": "int8_kv_f32_activation_cpu",
}
LOGIT_THRESHOLDS = {"max_abs": 0.1, "rmse": 0.1, "min_cosine": 0.99}


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def _exact_fields(value: Any, fields: tuple[str, ...], label: str) -> None:
    if not isinstance(value, dict) or tuple(value) != fields:
        raise ValueError(f"{label} fields/order must be exact")


def _int(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    return [_int(item, label, 0) for item in value]


def _hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be {length} lowercase hex characters")
    return value


def _relative_artifact(value: Any, suffix: str, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != suffix:
        raise ValueError(f"{label} must be a repository-relative {suffix.upper()[1:]}")
    return path


def _validate_named_cases(cases: Any, fields: tuple[str, ...], order: tuple[str, ...], label: str) -> None:
    if not isinstance(cases, list) or len(cases) != len(order):
        raise ValueError(f"{label} count must be exact")
    if tuple(case.get("name") if isinstance(case, dict) else None for case in cases) != order:
        raise ValueError(f"{label} order must be exact")
    for case in cases:
        _exact_fields(case, fields, label)


def validate_contract(data: Any) -> dict[str, Any]:
    _exact_fields(data, TOP_FIELDS, "top-level")
    if data["schema"] != "qx-unicode-chat-greedy-matrix-v1":
        raise ValueError("unsupported schema")
    if _int(data["issue"], "issue") != 22:
        raise ValueError("issue must be 22")

    source = data["source"]
    _exact_fields(source, SOURCE_FIELDS, "source")
    _relative_artifact(source["model_path"], ".gguf", "model_path")
    _relative_artifact(source["qxf_path"], ".qxf", "qxf_path")
    _hex(source["model_sha256"], 64, "model_sha256")
    _hex(source["qxf_sha256"], 64, "qxf_sha256")
    _hex(source["llama_cpp_commit"], 40, "llama_cpp_commit")
    _int(source["model_size"], "model_size", 1)
    _int(source["qxf_size"], "qxf_size", 1)
    _exact_fields(source["runtime"], RUNTIME_FIELDS, "runtime")
    if source["runtime"] != EXPECTED_RUNTIME:
        raise ValueError("runtime modalities do not match the fixed experiment")
    if source["oracle_args"] != ORACLE_ARGS:
        raise ValueError("oracle_args do not match fixed oracle commands")

    canonical = json.loads(TOKENIZER_ORACLE.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    if source["llama_cpp_commit"] != canonical.get("llama_cpp_commit"):
        raise ValueError("llama_cpp_commit does not match canonical tokenizer oracle")
    if source["model_sha256"] != canonical.get("source_model_sha256"):
        raise ValueError("model_sha256 does not match canonical tokenizer oracle")
    if source["model_size"] != canonical.get("source_model_size"):
        raise ValueError("model_size does not match canonical tokenizer oracle")
    if source["oracle_revision"] != f"llama.cpp@{source['llama_cpp_commit']}":
        raise ValueError("oracle_revision does not match llama_cpp_commit")

    _validate_named_cases(data["tokenizer_cases"], TOKENIZER_FIELDS, TOKENIZER_ORDER, "tokenizer_cases")
    for case in data["tokenizer_cases"]:
        if not isinstance(case["text"], str) or not isinstance(case["parse_special"], bool):
            raise ValueError("tokenizer case types invalid")
        _int_list(case["expected_token_ids"], "expected_token_ids")
        if case["expected_decode_text"] != case["text"]:
            raise ValueError("expected_decode_text must equal source text")
        if case["oracle_origin"] != "llama-tokenize" or case["mode"] != "vocab_only_cpu":
            raise ValueError("tokenizer oracle origin/mode invalid")

    _validate_named_cases(data["chat_cases"], CHAT_FIELDS, CHAT_ORDER, "chat_cases")
    for case in data["chat_cases"]:
        if not isinstance(case["messages"], list) or not case["messages"] or not isinstance(case["add_generation_prompt"], bool) or not isinstance(case["expected_text"], str):
            raise ValueError("chat case types invalid")
        _int_list(case["expected_token_ids"], "chat expected_token_ids")
        for message in case["messages"]:
            _exact_fields(message, MESSAGE_FIELDS, "chat message")
            if message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
                raise ValueError("chat message invalid")
        if case["oracle_origin"] != "llama_chat_apply_template" or case["mode"] != "vocab_only_cpu":
            raise ValueError("chat oracle origin/mode invalid")

    _validate_named_cases(data["greedy_cases"], GREEDY_FIELDS, GREEDY_ORDER, "greedy_cases")
    for case in data["greedy_cases"]:
        _int_list(case["prompt_token_ids"], "prompt_token_ids")
        steps = _int(case["generation_steps"], "generation_steps", 1)
        for field in ("expected_llama_f16_tokens", "expected_llama_q8_0_tokens", "expected_qx_tokens"):
            if len(_int_list(case[field], field)) != steps:
                raise ValueError("greedy sequence length mismatch")
        if case["oracle_origin"] != "llama_sequence_oracle" or case["mode"] != "llama.cpp:f16_cpu+q8_0_cpu vs qx:int8_kv+f32_activation_cpu":
            raise ValueError("greedy oracle origin/mode invalid")

    _exact_fields(data["claims"], CLAIM_FIELDS, "claims")
    if data["claims"] != {
        "tokenizer_parity": "case_local",
        "chat_template_render": "case_local",
        "greedy_sequence": "case_and_modality_local",
        "logit_parity": "not_claimed",
        "semantic_equivalence": "not_claimed",
        "global_modality_equivalence": "not_claimed",
    }:
        raise ValueError("claims exceed the fixed case-local contract")
    return data


def load_contract(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("contract is not valid UTF-8") from exc
    return validate_contract(json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant))


def _invoke(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, encoding="utf-8", capture_output=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {result.stderr[-2000:]}")
    return json.loads(result.stdout, object_pairs_hook=_pairs, parse_constant=_constant)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_artifact(root: Path, relative: str, size: int) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"artifact missing, external, or wrong size: {relative}")
    return path


def compare_case_logits(
    qx_dir: Path,
    llama_dirs: dict[str, Path],
    generation_steps: int,
) -> list[dict[str, Any]]:
    if set(llama_dirs) != {"f16", "q8_0"} or generation_steps <= 0:
        raise ValueError("case logit modalities/steps invalid")
    rows = []
    for step in range(generation_steps):
        qx_logits = qx_dir / f"step-{step}-logits.f32"
        f16_logits = llama_dirs["f16"] / f"step-{step}-logits.f32"
        q8_logits = llama_dirs["q8_0"] / f"step-{step}-logits.f32"
        f16_metrics = compare_logit_files(qx_logits, f16_logits, **LOGIT_THRESHOLDS)
        q8_metrics = compare_logit_files(qx_logits, q8_logits, **LOGIT_THRESHOLDS)
        for metrics, reference in ((f16_metrics, f16_logits), (q8_metrics, q8_logits)):
            metrics["qx"] = qx_logits.name
            metrics["llama"] = reference.name
        rows.append({
            "step": step,
            "llama_f16": f16_metrics,
            "llama_q8_0": q8_metrics,
        })
    return rows


def compare_activation_logits(
    qx_dirs: dict[str, Path],
    llama_dirs: dict[str, Path],
    generation_steps: int,
) -> dict[str, list[dict[str, Any]]]:
    if tuple(qx_dirs) != ("f32", "q8_k_compat"):
        raise ValueError("QX activation modes/order must be f32,q8_k_compat")
    return {
        activation: compare_case_logits(directory, llama_dirs, generation_steps)
        for activation, directory in qx_dirs.items()
    }


def run_contract(
    contract: dict[str, Any],
    root: Path,
    work: Path,
    run_greedy: bool = False,
    qx_activation_bisect: bool = False,
) -> dict[str, Any]:
    validate_contract(contract)
    root = root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    exe_path = root / "build/qxqxf.exe"
    qxt_path = root / "models/Qwen3-30B-A3B.qxt"
    exe = _require_artifact(root, "build/qxqxf.exe", exe_path.stat().st_size if exe_path.is_file() else -1)
    qxt = _require_artifact(root, "models/Qwen3-30B-A3B.qxt", qxt_path.stat().st_size if qxt_path.is_file() else -1)

    tokenizer_results = []
    for index, case in enumerate(contract["tokenizer_cases"]):
        prompt = work / f"tokenizer-{index}.txt"
        prompt.write_text(case["text"], encoding="utf-8", newline="")
        command = [str(exe), "tokenizer-encode", "--tokenizer", str(qxt), "--text-file", str(prompt)]
        if case["parse_special"]:
            command.append("--parse-special")
        actual_ids = _invoke(command)["token_ids"]
        if actual_ids != case["expected_token_ids"]:
            raise ValueError(f"tokenizer encode mismatch: {case['name']}")
        decoded = _invoke([str(exe), "tokenizer-decode", "--tokenizer", str(qxt), "--ids", ",".join(map(str, actual_ids))])
        if decoded["text"] != case["expected_decode_text"]:
            raise ValueError(f"tokenizer decode mismatch: {case['name']}")
        tokenizer_results.append({"name": case["name"], "token_ids": actual_ids, "decoded_text": decoded["text"], "status": "passed"})

    chat_results = []
    for case_index, case in enumerate(contract["chat_cases"]):
        command = [str(exe), "chat-template-render"]
        for message_index, message in enumerate(case["messages"]):
            content = work / f"chat-{case_index}-{message_index}.txt"
            content.write_text(message["content"], encoding="utf-8", newline="")
            command.extend(["--message", f"{message['role']}:{content}"])
        if case["add_generation_prompt"]:
            command.append("--add-generation-prompt")
        rendered = _invoke(command)
        if rendered["text"] != case["expected_text"] or rendered["utf8_bytes"] != len(case["expected_text"].encode("utf-8")):
            raise ValueError(f"chat-template bytes mismatch: {case['name']}")
        rendered_path = work / f"chat-rendered-{case_index}.txt"
        rendered_path.write_text(rendered["text"], encoding="utf-8", newline="")
        actual_ids = _invoke([str(exe), "tokenizer-encode", "--tokenizer", str(qxt), "--text-file", str(rendered_path), "--parse-special"])["token_ids"]
        if actual_ids != case["expected_token_ids"]:
            raise ValueError(f"chat-template token mismatch: {case['name']}")
        chat_results.append({"name": case["name"], "utf8_bytes": rendered["utf8_bytes"], "token_ids": actual_ids, "status": "passed"})

    greedy_report: dict[str, Any] = {"status": "not_run", "reason": "real-model experiment is a separate explicit gate"}
    if run_greedy:
        model = _require_artifact(root, contract["source"]["model_path"], contract["source"]["model_size"])
        qxf = _require_artifact(root, contract["source"]["qxf_path"], contract["source"]["qxf_size"])
        llama_oracle = ROOT / "build" / "llama_sequence_oracle.exe"
        if not llama_oracle.is_file():
            raise ValueError("llama_sequence_oracle.exe is missing; build the reference oracles first")
        if _sha256(model) != contract["source"]["model_sha256"]:
            raise ValueError("model_sha256 does not match the GGUF artifact")
        if _sha256(qxf) != contract["source"]["qxf_sha256"]:
            raise ValueError("qxf_sha256 does not match the QXF artifact")
        greedy_results = []
        for case in contract["greedy_cases"]:
            if len(case["prompt_token_ids"]) != 1:
                raise ValueError("QX state-loop gate requires one prompt token per case")
            case_work = work / f"greedy-{case['name']}"
            activations = ("f32", "q8_k_compat") if qx_activation_bisect else ("f32",)
            qx_logits = {activation: case_work / f"qx-{activation}" for activation in activations}
            llama_logits = {kv: case_work / f"llama-{kv}" for kv in ("f16", "q8_0")}
            for directory in qx_logits.values():
                directory.mkdir(parents=True)
            for directory in llama_logits.values():
                directory.mkdir()
            qx_tokens = {}
            for activation in activations:
                payload = _invoke([
                    str(exe), "state-loop-probe", "--in", str(qxf), "--prompt-token", str(case["prompt_token_ids"][0]),
                    "--steps", str(case["generation_steps"]), "--layers", "48", "--ctx", "4", "--kv", "int8",
                    "--activation", activation, "--temperature", "0", "--seed", "7", "--full-moe", "--final-head", "--top-n", "5",
                    "--dump-residuals", str(qx_logits[activation]),
                ])
                actual = [step["selected_token"] for step in payload["tokens"]]
                if actual != case["expected_qx_tokens"]:
                    raise ValueError(f"greedy QX {activation} mismatch: {case['name']}")
                qx_tokens[activation] = actual
            llama_tokens = {}
            for kv in ("f16", "q8_0"):
                llama_payload = _invoke([
                    str(llama_oracle), str(model), str(case["prompt_token_ids"][0]),
                    str(case["generation_steps"]), kv, str(llama_logits[kv]),
                ])
                expected = case[f"expected_llama_{kv}_tokens"]
                if llama_payload.get("llama_commit") != contract["source"]["llama_cpp_commit"]:
                    raise ValueError("llama.cpp oracle commit mismatch")
                if llama_payload.get("generated_tokens") != expected:
                    raise ValueError(f"greedy llama {kv} mismatch: {case['name']}")
                llama_tokens[kv] = llama_payload["generated_tokens"]
            activation_comparisons = {
                activation: compare_case_logits(directory, llama_logits, case["generation_steps"])
                for activation, directory in qx_logits.items()
            }
            greedy_results.append({
                "name": case["name"],
                "llama_f16_tokens": llama_tokens["f16"],
                "llama_q8_0_tokens": llama_tokens["q8_0"],
                "qx_tokens": qx_tokens["f32"],
                "logit_comparisons": activation_comparisons["f32"],
                "qx_activation_bisect": {
                    activation: {
                        "qx_tokens": qx_tokens[activation],
                        "logit_comparisons": comparisons,
                    }
                    for activation, comparisons in activation_comparisons.items()
                },
                "modalities_reported_separately": True,
                "status": "passed",
            })
        threshold_results = [
            row[modality]["pass"]
            for case in greedy_results
            for row in case["logit_comparisons"]
            for modality in ("llama_f16", "llama_q8_0")
        ]
        greedy_report = {
            "status": "passed",
            "passed": len(greedy_results),
            "logit_thresholds": LOGIT_THRESHOLDS,
            "logit_threshold_summary": {
                "comparisons": len(threshold_results),
                "passed": sum(threshold_results),
                "failed": len(threshold_results) - sum(threshold_results),
                "claim": "case_local_measurement_only",
            },
            "activation_threshold_summary": {
                activation: {
                    "comparisons": sum(
                        len(case["qx_activation_bisect"][activation]["logit_comparisons"]) * 2
                        for case in greedy_results
                    ),
                    "passed": sum(
                        row[modality]["pass"]
                        for case in greedy_results
                        for row in case["qx_activation_bisect"][activation]["logit_comparisons"]
                        for modality in ("llama_f16", "llama_q8_0")
                    ),
                }
                for activation in activations
            },
            "cases": greedy_results,
        }

    return {
        "schema": "qx-unicode-chat-greedy-report-v1",
        "source": contract["source"],
        "tokenizer": {"passed": len(tokenizer_results), "cases": tokenizer_results},
        "chat_template": {"passed": len(chat_results), "cases": chat_results},
        "greedy": greedy_report,
        "overall_status": "passed" if run_greedy else "partial_environmental",
        "claims": contract["claims"],
    }


def publish_report(path: Path, report: Any) -> None:
    if path.exists():
        raise ValueError("report path already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ValueError("report path already exists") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-greedy", action="store_true")
    parser.add_argument("--qx-activation-bisect", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    with tempfile.TemporaryDirectory(prefix="qx-unicode-chat-") as temp:
        report = run_contract(
            contract,
            args.root.resolve(),
            Path(temp),
            run_greedy=args.run_greedy,
            qx_activation_bisect=args.qx_activation_bisect,
        )
    publish_report(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
