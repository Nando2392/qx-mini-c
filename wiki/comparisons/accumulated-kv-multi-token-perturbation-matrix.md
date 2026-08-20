---
title: Accumulated KV Multi-Token Perturbation Matrix
created: 2026-08-20
updated: 2026-08-20
type: comparison
tags: [qwen3-moe, kv, snapshot, replay, residual, perturbation, multi-token]
sources: [state-loop-probe, issue-21]
confidence: high
---

# Accumulated KV multi-token perturbation matrix

Issue: [#21](https://github.com/Nando2392/qx-mini-c/issues/21)

## Scientific question

With a prefix of at least two accumulated positions held fixed by the verified [[accumulated-kv-snapshot-replay]] seam, how does one controlled residual perturbation at the next position propagate through the remaining suffix, and does that response differ between KV F16 and INT8?

This extends [[scaled-residual-token-modality-matrix]] beyond position zero. It does not use another runtime as an oracle and does not assume F16/INT8 numerical equivalence.

## Fixed contract

Each KV mode is an independent cell with the same model, producer binary, revision, initial token, prefix length, continuation length, activation format, layer boundary, context, seed and ordered signed scales.

For each cell the runner must:

1. execute an uninterrupted `prefix_steps + 1` baseline;
2. capture the accumulated K/V state after `prefix_steps`;
3. validate the snapshot payload, model and producer provenance, geometry, hashes and accumulated token IDs;
4. replay exactly one continuation step from the recorded next token;
5. require exact direct-baseline/replay parity before interpreting any perturbation;
6. derive a deterministic synthetic direction by changing one recorded finite F32 residual coordinate by `direction_magnitude`;
7. inject every ordered scale from the same validated snapshot and residual boundary;
8. require scale zero to reproduce the unmodified replay exactly;
9. emit a complete report only after all F16 and INT8 cells pass.

The perturbation is deliberately synthetic and controlled. It measures causal response to a recorded direction; it is not an oracle error vector. Runtime mode `full_moe` means that the native execution evaluates the full MoE path; it does **not** mean that the synthetic coordinate direction came from a full-MoE oracle or error vector.

## Fail-closed boundary

The planner and analyzer reject:

- incomplete, duplicated, extra or reordered KV/scales/cells;
- JSON booleans in integer fields, integral floats where exact integers are required, duplicate keys and NaN/Inf;
- non-finite scales, direction values or metrics;
- changed artifact paths, sizes or SHA-256 values;
- model, producer revision, activation, seed, token sequence, continuation token or geometry mismatches;
- partial runtime results, missing layer coverage or inconsistent routing counts;
- snapshot truncation, extension, same-length mutation or embedded digest mismatch;
- a scale-zero run that differs from the unmodified replay.

One failed cell invalidates the whole matrix. Runtime manifests remain machine-local because they contain absolute paths. The analyzer strips paths from its report envelope, retains artifact SHA-256 provenance, and the accredited report is versioned under [`wiki/evidence/issue-21-full-moe-report.json`](../evidence/issue-21-full-moe-report.json).

## Metrics and claims

Per run, the report records the effective F32 input delta, final-residual delta from scale zero, selected token and whether ordered top-k routing changed. The `cross_mode` section aligns runs by ordinal and scale and records selected-token parity, routing-change agreement, and each mode's final L2 without collapsing the independent F16/INT8 cells.

Allowed claims are limited to exact replay control, selected-token parity or divergence, measured finite residual deltas, observed routing transitions, and differences between the tested F16/INT8 cells.

The experiment must not claim llama.cpp parity, universal model behavior, numerical equivalence of KV formats, a production threshold change, or model-quality improvement. Absence of a transition at tested scales is not proof of global linearity or stability.

## Reproduction

The canonical runner is `scripts/accumulated_kv_perturbation_matrix.py`, with three fail-closed commands:

- `plan` validates dimensions, provenance and deterministic ordering without running the model;
- `run` executes all cells, writes `matrix-manifest.json`, revalidates every artifact and writes `report.json` only after the complete matrix passes;
- `analyze` independently revalidates trusted model/binary/tokenizer hashes, native snapshot v2 contents, runtime JSON, residual formula, ordering and scale-zero controls.

The accredited local experiment uses an isolated MSVC build from `0fc21c697994782b1f393dabd198855ef5ab939f`, the real `Qwen3-30B-A3B-UD-IQ2_M.qxf` fixture, token `42`, a two-position prefix, one continuation, layers `0..1`, injection at layer `1`, residual width `2048`, scales `[-1,0,1]`, seed `7`, and KV F16/INT8. Machine-local runtime artifacts remain under `experiment-run-real3` outside Git. The path-free analyzer report is versioned with SHA-256 `38d7f2a976e306a536c861f5b66cf9a6f61eaed1e4d9c7a46a390c1b596e45b6`.

Observed pre-publication evidence:

- direct baseline and accumulated snapshot/replay controls are exact in both KV cells;
- both scale-zero runs have exact zero input/final deltas and unchanged routing;
- all six perturbation runs select token `56`;
- F16 scale `+1` changes ordered routing; F16 `-1` and both INT8 signed runs do not;
- non-zero final L2 ranges from `0.1380961360` to `0.1406689282` in this tested slice.
- cross-mode selected-token parity holds at all three scales; routing-change agreement differs only at scale `+1`.

These observations are bounded to the recorded fixture and direction. They do not establish semantic causality, global stability, or F16/INT8 equivalence.

## Status

Implementation and the real local experiment are GREEN under Issue #21. Publication remains gated on the final immutable payload digest, full regression/lint gates, independent reviews and GitHub CI.

Related roadmap: [[current-status-and-roadmap]].
