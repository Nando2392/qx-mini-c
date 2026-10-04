#!/usr/bin/env python
"""Bounded Issue 87 F32 matrix. No builds or performance acceptance claims."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
from pathlib import Path
import expert_cache_acceptance as base
from expert_cache_logit_bytes import FullLogitBytesError, require_exact_policy_bytes

AcceptanceError = base.AcceptanceError
THREAD_POLICIES = ("serial", "moe-pool")
MAX_RSS_BYTES = 2 * 1024**3


def compact_run(raw, thread_policy, policy, budget, capture_dir=None, evidence_root=None):
    sampled = base.exact_int(raw.get("sampled_peak_rss_bytes"), "sampled RSS", 0)
    final = base.exact_int(raw.get("final_peak_working_set_bytes"), "final working-set peak", 1)
    if sampled > MAX_RSS_BYTES or final > MAX_RSS_BYTES:
        raise AcceptanceError("root working set exceeded 2 GiB")
    # #86 requires a positive sample. Validate the zero-sample case above with
    # retained-handle proof, adapting only its input, never relabeling the peak.
    base_raw = {**raw, "sampled_peak_rss_bytes": sampled or final}
    compact = base.compact_run(base_raw, "f32", policy, budget, capture_dir, evidence_root)
    compact["sampled_peak_rss_bytes"] = sampled
    compact["final_peak_working_set_bytes"] = final
    if compact["forward_positions"] != 2:
        raise AcceptanceError("exactly two forward positions required")
    if raw.get("returncode") != 0 or raw.get("timed_out") is not False:
        raise AcceptanceError("unsuccessful native process")
    # The default serial payload is unchanged. Validate its existing execution
    # counters rather than requiring fields introduced only by the opt-in policy.
    if thread_policy == "serial":
        legacy = base.object_value(raw["payload"].get("thread_profile"), "thread_profile")
        if (legacy.get("policy") != "serial"
                or base.exact_int(legacy.get("workers_used"), "workers_used") != 1
                or base.exact_int(legacy.get("parallel_jobs"), "parallel_jobs") != 0):
            raise AcceptanceError("serial control must execute serial with zero parallel jobs")
        profile = raw["payload"].get("moe_pool_profile", {"workers": 0, "jobs": 0})
    else:
        profile = raw["payload"].get("moe_pool_profile")
    profile = base.object_value(profile, "moe_pool_profile")
    workers = base.exact_int(profile.get("workers"), "moe_pool_profile.workers")
    jobs = base.exact_int(profile.get("jobs"), "moe_pool_profile.jobs")
    if thread_policy == "moe-pool":
        if workers != 2 or jobs <= 0:
            raise AcceptanceError("moe-pool requires two actual workers and positive jobs")
    elif thread_policy != "serial" or workers != 0 or jobs != 0:
        raise AcceptanceError("serial requires zero MoE workers/jobs")
    compact["moe_pool_profile"] = profile
    return compact


def command_for(executable, model, tokenizer, prompt, dump_dir, thread_policy, policy, budget):
    if thread_policy not in THREAD_POLICIES:
        raise AcceptanceError("unsupported thread policy")
    command = base.command_for(executable, model, tokenizer, prompt, dump_dir, "f32", policy, budget)
    return command + ["--thread-policy", thread_policy, "--threads", "2" if thread_policy == "moe-pool" else "1"]


def matrix_plan():
    return [(thread, policy, phase, rep + 1)
            for thread in THREAD_POLICIES for policy in base.POLICIES
            for phase, count in (("warmup", 1), ("measured", 2))
            for rep in range(count)]


def final_peak_working_set(process):
    """Query the retained creation handle, never reopen a potentially reused PID.

    CPython's Windows Popen owns _handle until finalization, including after
    wait/poll. Win32 keeps that process object alive while a handle is retained;
    GetProcessMemoryInfo's PeakWorkingSetSize is its lifetime working-set peak.
    References: Microsoft /windows/win32/procthread/terminating-a-process and
    /windows/win32/api/psapi/{nf-psapi-getprocessmemoryinfo,
    ns-psapi-process_memory_counters}. This certifies the root, not descendants.
    """
    import ctypes
    from ctypes import wintypes

    if sys.platform != "win32":
        raise OSError("final lifetime RSS proof requires Windows")

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not query(process._handle, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    peak = int(counters.PeakWorkingSetSize)
    if peak <= 0:
        raise OSError("final lifetime working-set peak is not positive")
    return peak


def run_one(command, run_dir, cwd):
    """Supervise a threads-only root, not a lifetime process tree.

    Samples permit early termination; retained-handle lifetime working-set peak
    permits post-exit rejection. Neither is a hard OS memory allocation limit.
    """
    started = time.monotonic()
    record = {"argv": command, "cwd": str(cwd), "started_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": 300, "rss_limit_bytes": MAX_RSS_BYTES,
              "timed_out": False, "rss_limit_exceeded": False, "sampled_peak_rss_bytes": 0,
              "rss_monitoring_error": None,
              "rss_scope": "root_process_working_set", "hard_os_memory_limit": False,
              "lifetime_tree_guarantee": False,
              "rss_enforcement": "sampled_kill_and_post_exit_peak_rejection"}
    stdout_path, stderr_path = run_dir / "stdout.json", run_dir / "stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr)
        record["pid"] = process.pid
        try:
            base.atomic_json(run_dir / "invocation.json", record)
            phase = "initialize"
            children = []
            exit_race = False
            try:
                try:
                    tracked = base.psutil.Process(process.pid)
                    while process.poll() is None:
                        phase = "descendants"
                        children = tracked.children(recursive=True)
                        if children:
                            # Native driver uses CreateThread, never subprocesses.
                            # Enumeration is only an observed-contract check, not
                            # proof that no short-lived descendant ever existed.
                            record["forbidden_child_pids"] = [child.pid for child in children]
                            raise OSError(f"root driver contract violation: forbidden child processes {record['forbidden_child_pids']}")
                        phase = "sample"
                        rss = tracked.memory_info().rss
                        record["sampled_peak_rss_bytes"] = max(record["sampled_peak_rss_bytes"], rss)
                        record["timed_out"] = time.monotonic() - started >= 300
                        record["rss_limit_exceeded"] = record["sampled_peak_rss_bytes"] > MAX_RSS_BYTES
                        if record["timed_out"] or record["rss_limit_exceeded"]:
                            phase = "cleanup"
                            base.stop_tree(process)
                            break
                        phase = "wait"
                        time.sleep(base.RSS_SAMPLE_INTERVAL_SECONDS)
                except base.psutil.NoSuchProcess as exc:
                    # children() also reopens the root and can race its exit.
                    # Only an identified, successful root has retained-handle proof.
                    if (phase not in ("initialize", "descendants", "sample")
                            or exc.pid != process.pid or process.poll() != 0):
                        raise
                    exit_race = True
                if not record["timed_out"] and not record["rss_limit_exceeded"]:
                    phase = "final_peak"
                    peak = final_peak_working_set(process)
                    if type(peak) is not int or peak <= 0:
                        raise OSError("final lifetime working-set peak is not a positive integer")
                    record["final_peak_working_set_bytes"] = peak
                    record["rss_limit_exceeded"] = peak > MAX_RSS_BYTES
                    record["rss_exit_race_recovered"] = exit_race
                    record["timed_out"] = time.monotonic() - started >= 300
            except (base.psutil.Error, OSError) as exc:
                # Initial/live/descendant faults and failed final proof remain
                # fail-closed; a prior positive sample is never a substitute.
                record["rss_monitoring_error"] = {
                    "type": type(exc).__name__, "message": str(exc),
                    "pid": getattr(exc, "pid", process.pid), "phase": phase,
                    "observed_utc": datetime.now(timezone.utc).isoformat()}
                base.atomic_json(run_dir / "process.json", record)
                # The root may already have exited: preserve enumerated identities
                # rather than relying on stop_tree to rediscover orphaned children.
                for child in children:
                    try:
                        child.kill()
                    except base.psutil.NoSuchProcess:
                        pass
                _, alive = base.psutil.wait_procs(children, timeout=5)
                if alive:
                    raise AcceptanceError(f"forbidden child cleanup failed: {[child.pid for child in alive]}")
                base.stop_tree(process)
        finally:
            try:
                if process.poll() is None:
                    base.stop_tree(process)
            finally:
                record.update(returncode=process.returncode, wall_seconds=time.monotonic() - started,
                              finished_utc=datetime.now(timezone.utc).isoformat())
                base.atomic_json(run_dir / "process.json", record)
    record["stdout"] = base.artifact(stdout_path, run_dir.parent)
    record["stderr"] = base.artifact(stderr_path, run_dir.parent)
    base.atomic_json(run_dir / "process.json", record)
    if record["rss_monitoring_error"] is not None:
        raise AcceptanceError(f"RSS monitoring failed: {record['rss_monitoring_error']}")
    if record["timed_out"] or record["rss_limit_exceeded"] or record["returncode"] != 0:
        raise AcceptanceError(f"native process failed: {record['returncode']}; timeout={record['timed_out']}; RSS wall={record['rss_limit_exceeded']}")
    record["payload"] = base.object_value(json.loads(stdout_path.read_text(encoding="utf-8")), "native payload")
    return record


def equivalence(cells):
    by_key = {(c["thread_policy"], c["policy"]): c for c in cells}
    expected = {(t, p) for t in THREAD_POLICIES for p in base.POLICIES}
    if len(cells) != 4 or set(by_key) != expected:
        raise AcceptanceError("missing or duplicate matrix cells")
    reference_cell = by_key[("serial", "none")]
    root = reference_cell["evidence_root"]
    ordered = [reference_cell] + [c for c in cells if c is not reference_cell]
    runs = []
    policy_runs = {policy: [] for policy in base.POLICIES}
    for cell in ordered:
        if cell.get("evidence_root") != root or len(cell["warmups"]) != 1 or len(cell["measured"]) != 2:
            raise AcceptanceError("matrix run count/evidence root mismatch")
        cell_runs = cell["warmups"] + cell["measured"]
        runs.extend(cell_runs)
        policy_runs[cell["policy"]].extend(cell_runs)
    reference = (runs[0]["generated_token_ids"], runs[0]["full_logits_checksums"])
    if reference[0] != base.EXPECTED_F32_TOKENS or any(
        (r["generated_token_ids"], r["full_logits_checksums"]) != reference for r in runs):
        raise AcceptanceError("exact tokens/checksums differ from serial control")
    if any(not isinstance(r.get("full_logit_artifacts"), list) for r in runs):
        raise AcceptanceError("missing full-logit byte artifacts")
    try:
        return require_exact_policy_bytes(
            [r["full_logit_artifacts"] for r in policy_runs["none"]],
            [r["full_logit_artifacts"] for r in policy_runs["resident-packed"]],
            Path(root))
    except FullLogitBytesError as exc:
        raise AcceptanceError(str(exc)) from exc


def freeze(root, executable, model, tokenizer):
    frozen = base.freeze(root, executable, model, tokenizer)
    header = "src/qx_row_pool.h"
    if not any(p["path"] == header for p in frozen["source_files"]):
        raise AcceptanceError("required row-pool header missing from source freeze")
    frozen["issue87_helpers"] = [base.artifact(root / p, root) for p in (
        "scripts/moe_pool_acceptance.py", "tests/test_moe_pool_acceptance.py")]
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description="Issue 87 bounded F32 correctness acceptance; no build")
    parser.add_argument("--driver-exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    executable, model, tokenizer, output = [base.resolve(root, p) for p in (
        args.driver_exe, args.model, args.tokenizer, args.output_dir)]
    created = False
    pre = None
    try:
        if sys.platform != "win32":
            raise AcceptanceError("moe-pool acceptance requires Windows")
        if not base.portable_path(output, root).startswith("build/issue87-") or output.exists():
            raise AcceptanceError("output must be fresh build/issue87-*; no overwrite")
        for path in (executable, model, tokenizer):
            if not path.is_file():
                raise AcceptanceError(f"missing required input: {path}")
            base.portable_path(path, root)
        output.mkdir(parents=True, exist_ok=False)
        created = True
        budget = base.derive_budget(base.parse_qxf(model), int(base.psutil.virtual_memory().available))
        base.atomic_json(output / "budget.json", budget)
        if not budget["feasible"]:
            raise AcceptanceError("metadata-derived packed budget infeasible")
        prompt = output / "prompt.txt"
        prompt.write_text(base.PROMPT_TEXT, encoding="utf-8", newline="")
        pre = freeze(root, executable, model, tokenizer)
        pre["prompt"] = base.artifact(prompt, root)
        base.atomic_json(output / "inputs-pre.json", pre)
        cells = []
        for thread in THREAD_POLICIES:
            for policy in base.POLICIES:
                cell = {"thread_policy": thread, "policy": policy, "activation": "f32",
                        "evidence_root": str(output), "warmups": [], "measured": []}
                for index, (t, p, phase, rep) in enumerate(matrix_plan(), 1):
                    if (t, p) != (thread, policy):
                        continue
                    run_dir = output / f"run-{index:02d}-{thread}-{policy}-{phase}-{rep}"
                    run_dir.mkdir(exist_ok=False)
                    command = command_for(executable, model, tokenizer, prompt, run_dir,
                                          thread, policy, budget["budget_bytes"])
                    raw = run_one(command, run_dir, root)
                    compact = compact_run(raw, thread, policy, budget["budget_bytes"], run_dir, output)
                    compact["process"] = {k: v for k, v in raw.items() if k != "payload"}
                    cell["warmups" if phase == "warmup" else "measured"].append(compact)
                cell["summary"] = base.summary(cell["measured"])
                cells.append(cell)
        equality = equivalence(cells)
        post = freeze(root, executable, model, tokenizer)
        post["prompt"] = base.artifact(prompt, root)
        base.atomic_json(output / "inputs-post.json", post)
        if pre != post:
            raise AcceptanceError("frozen sources/executable/model/tokenizer/prompt changed")
        report = {"schema": "qx-issue87-moe-pool-real-ab-v1", "issue": 87,
                  "status": "pass", "performance_pass": False, "timing_descriptive_only": True,
                  "fixed_contract": {"prompt": base.PROMPT_TEXT, "total_processes": 12,
                                     "forward_positions_per_run": 2, "warmups_per_cell": 1,
                                     "measured_per_cell": 2, "timeout_seconds": 300,
                                     "rss_limit_bytes": MAX_RSS_BYTES, "pool_workers": 2,
                                     "rss_scope": "root_process_working_set",
                                     "hard_os_memory_limit": False,
                                     "lifetime_tree_guarantee": False,
                                     "child_process_contract": "forbidden_if_observed",
                                     "rss_enforcement": "sampled_kill_and_post_exit_peak_rejection"},
                  "qxf_budget_derivation": budget, "inputs_pre": pre, "inputs_post": post,
                  "cells": cells, "exact_output_equivalence": equality}
        base.atomic_json(output / "report.json", report)
        print(json.dumps({"status": "pass", "report": base.portable_path(output / "report.json", root)}))
        return 0
    except (AcceptanceError, FullLogitBytesError, OSError, ValueError, subprocess.SubprocessError) as exc:
        if created:
            failure = {"status": "failed_closed", "error": str(exc), "performance_pass": False}
            if pre is not None:
                try:
                    failure["inputs_post"] = freeze(root, executable, model, tokenizer)
                except (AcceptanceError, OSError, subprocess.SubprocessError) as post_exc:
                    failure["post_freeze_error"] = str(post_exc)
            base.atomic_json(output / "failure.json", failure)
        print(f"Issue 87 acceptance failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
