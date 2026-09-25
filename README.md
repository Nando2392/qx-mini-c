# QX-mini-MoE

Experimental C runtime and mmap-oriented model format for correctness-first local inference of **Qwen3-30B-A3B MoE**.

> Status: research runtime. Qwen3 GPT-2/Qwen2 BPE parity is GREEN for fixed ASCII, Unicode, whitespace and ChatML prompts. QXF1 now rejects malformed manifests, directories, dimensions, placements, overlaps and legacy truncated rows fail-closed. The C loop prefills IDs, then re-embeds greedy outputs with persistent per-layer INT8 KV across all 48 layers and the complete 151936-row Q6_K head. An opt-in CUDA final-head path has point-in-time functional evidence, but its release gates are not passed. Exhaustive Unicode/chat-template coverage and external end-to-end residual parity are not finished. Probe timing is not conversational decode throughput.

## Goals

- Own C runtime rather than wrapping an inference framework.
- Convert a real GGUF into the QXF1 tensor-copy container.
- Validate every numerical slice independently before optimization.
- Run Qwen3-30B-A3B on constrained RAM/VRAM using expert slicing and caching.
- Keep Python outside the final inference path; use it for tests and references.

## Implemented path

```text
text prompt
→ QXT2 tokenizer sidecar: Qwen2 pre-tokenizer + GPT-2 byte BPE
→ exact token IDs
→ GGUF tensor bytes copied into QXF1
→ QXF1 loader/checksum
→ embedding Q4_K
→ attention RMSNorm
→ Q/K/V IQ4_XS
→ per-head Q/K RMSNorm
→ Qwen3 split-half RoPE
→ dynamic INT8 KV
→ GQA 32Q/4KV + causal softmax
→ attention output 4096→2048
→ attention residual + FFN RMSNorm
→ F32 router + top-8
→ IQ2_XS gate/up + SwiGLU
→ IQ3_XXS down
→ weighted MoE sum + residual propagation across layers 0–47
→ final RMSNorm F32
→ output.weight Q6_K, 2048→151936
→ complete logits + argmax/top-N
→ selected token embedding at the next position
→ persistent KV update and repeated 48-layer forward
```

The fixed-token QX greedy gate is GREEN for `42 → 1124 → 50853` after the Q5_K decoder/attention fix. Fixed-prompt tokenizer parity against `llama-tokenize` is also GREEN. `Hello!` maps to `[9707, 0]`, prefills both positions and generates `[358, 1184]`. These two sequences exactly match the pinned llama.cpp F16 oracle; broader exact residual/logit and prompt coverage remains gated.

`scripts/compare_logits.py` exposes the same full-sidecar metrics through `compare_logit_files(...)` and its existing CLI. This reusable seam is preparation for case-local parity matrices; it does not itself run a model, expand coverage or claim global logit equivalence.

The real CPU matrix now records complete logits for three fixed-token cases over two generation steps, comparing QX separately with llama.cpp F16 and Q8_0 KV. All 12 argmax comparisons match, while all 12 comparisons fail the configured full-logit thresholds (`max_abs <= 0.1`, `RMSE <= 0.1`, cosine `>= 0.99`). This is case-local greedy agreement and a numeric-parity refutation, not global logit or semantic equivalence.

The activation bisect localizes the first material amplification to the layer-1 MoE: the layer-input delta grows about 35x across the MoE while routing top-8 remains exact. Replaying the exact llama F16 `ffn_inp` keeps the F32 MoE error (`RMSE 0.0345051`) but closes it under `q8_k_compat` (`RMSE 9.62582e-7`). The repeated real matrix preserves 3/3 greedy and 12/12 argmax for both QX modes; Q8_K improves full-logit thresholds from 0/12 to 1/12, including token-42 step 0 vs llama Q8_0 (`max_abs 0.0997415`, `RMSE 0.0220756`, cosine `0.9999756`). This is a causal, case-local diagnostic—not parity, semantic equivalence, or permission to promote defaults. Evidence: `wiki/evidence/issue-76-first-divergence-localization.json` and `wiki/evidence/issue-76-activation-parity-bisect-report.json`.

The accumulated-KV cross-activation bisect reuses the published snapshot seam instead of rebuilding it. Across three cases it executes 12 cells (`prefix activation × continuation activation`) with INT8 KV; all six diagonal capture/replay controls are byte-exact. None of the 24 QX-vs-llama full-logit comparisons pass the unchanged thresholds, while 22/24 preserve argmax. The effect is token-dependent interaction rather than one globally dominant axis: token 42 and token 56 preserve their continuation argmax in all four cells, but token 1000 flips from `67075` to `1318` only for an F32-produced prefix snapshot consumed by a Q8_K continuation. This narrows the next gate to current-step residual/routing under that fixed snapshot; it does not justify a kernel fix or default promotion. Evidence: `wiki/evidence/issue-77-cross-activation-localization.json` and `wiki/evidence/issue-77-accumulated-kv-cross-activation-report.json`.

The fixed-snapshot residual replay now holds token 1000's F32-produced INT8 KV snapshot constant and resumes at the first cross-activation routing change, layer 2. Both F32 and Q8_K same-mode controls reproduce suffix routing, final residual, logits and selected token exactly. Injecting the exact F32 layer-1 residual into a Q8_K suffix still selects `1318`, not the F32 continuation's `67075`; layer-2 expert selection remains F32-exact and routing first departs from F32 at layer 3. The diagnostic logits pass neither unchanged full-logit comparison (`vs F32: max_abs 0.655345, RMSE 0.136190, cosine 0.997983`; `vs Q8_K: max_abs 0.949061, RMSE 0.290906, cosine 0.992844`). This localizes the next boundary to layer-2 output / layer-3 input for this case only; it is not global parity, semantic equivalence, a kernel root cause or permission to promote Q8_K. Evidence: `wiki/evidence/issue-78-fixed-kv-residual-bisect-report.json` and `wiki/evidence/issue-78-continuation-localization.json`.

The layer-3 seam experiment holds the exact F32 layer-2 output and F32-produced INT8 KV snapshot fixed while comparing F32 and `q8_k_compat`. The common input is byte-exact, but attention changes the FFN input (`max_abs 0.00958681`, RMSE `0.00130100`), after which integrated ordered routing ends in expert `89` for F32 and `22` for Q8_K. Replaying the same F32 FFN input through both MoE modes restores identical ordered IDs ending in `89`; expert outputs still differ (`max_abs 0.00210665`, RMSE `0.000659551`). Independently reconstructed opt-in `integrated_double` MoE and layer outputs are byte-exact controls in both modes, while the native default remains `legacy_f32`. Issue #80 is CLOSED in commit `b0c4019b493a2817b4c9d2219b917783266c77f9`; GitHub Actions run `35152297016` passed. This case-local result does not identify a kernel bug, establish global parity, promote a default or select the next experiment. Evidence: `wiki/evidence/issue-80-layer3-seams-report.json`.

Here, routing equality means exact equality of the ordered `selected_experts` IDs, not just set membership. It does not establish equal expert weights or equal numeric layer outputs. Layer 3 is the first observed ordered-ID difference after injection, not proof that the numeric error originates there.

## Native CPU generation (Issue #81)

The canonical native entry point is now a prompt-text CLI that loads the bound QXT2 tokenizer, runs the existing 48-layer F32/INT8-KV loop, and emits one JSON line:

```bash
python -c "open('issue-81-prompt.txt','wb').write(b'Hello!')"
build/qxqxf.exe generate --in models/Qwen3-30B-A3B-UD-IQ2_M.qxf --tokenizer models/Qwen3-30B-A3B.qxt --text-file issue-81-prompt.txt --max-tokens 2 --ctx 3
```

The parent-verified real-model acceptance result is prompt IDs `[9707, 0]` and deterministic generated IDs `[358, 1184]`, decoded through the same QXT sidecar. The public C API `qx_run_native_generation(...)` exposes the shared loop without routing generation through Python. Controlled API runs stop on token `358` when it is configured as EOS, stop on the second token for EOS `1184`, and return both tokens with `eos_token_id=-1`.

Supported limits are explicit: `max_tokens` and prompt count are non-zero and `ctx` is `1..4096`. The compatibility API `qx_run_native_generation(...)` retains its 64-position result contract; the caller-buffer API `qx_run_native_generation_into_with_options(...)` admits a forward-position budget up to 4096 and `<= ctx`. Admission to 4096 is not evidence of an actual 4K run, quality sweep or soak closure. The CLI binds vocabulary size, payload fingerprint, BOS, EOS and flags to the canonical Qwen3-30B-A3B QXT before model I/O. The split-UTF-8 case is a tokenizer decoder fixture only, not generate end-to-end coverage.

Issue #81 is closed in commit `0305290b3aba00d9db62f32caacfbc2e14cdbeb`; GitHub Actions run `35412907586` passed. This slice does not establish global model/logit parity, CUDA support, 4K runtime coverage or sustained throughput. Reproduction commands, exact gates and point-in-time source/test/executable hashes are in `wiki/evidence/issue-81-native-generation-report.json`.

## Native CPU policy measurement (Issue #82)

Issue #82 adds opt-in native-generation policies and an opt-in `--execution-profile` payload while preserving the default JSON contract and the compatibility C API. The policy surface is `--io-backend buffered|mmap`, `--scratch-policy ephemeral|persistent`, `--kernel-policy baseline|fused`, `--thread-policy serial|pool`, and `--threads N`; existing defaults remain buffered, ephemeral, baseline, serial, and one thread. Fusion and threading in this slice apply only to the final `output.weight` head, not attention or MoE.

The parent-verified real-model report has SHA-256 `ed801a0debc96d71dc7ea634cca434b37be0ed10024f84bb3bcfa2e53bda4125`. It ran 24 native CLI processes (four warmups plus five measured runs in each of four cells). Every run returned token IDs `[358,1184]`, text `" I need"`, and full-logit checksums `13347842135191822952` and `6249376751730758761`. Medians and median absolute deviations below are calculated over the five measured runs per cell:

| Cell | MSVC decode phase-local elapsed wall, s | End-to-end wall, s | Sampled peak RSS, MiB |
|---|---:|---:|---:|
| `baseline` | 21.280 ± 0.134 | 30.360 ± 0.173 | 19.422 ± 0.000 |
| `persistent_fused_serial1` | 22.565 ± 0.131 | 31.781 ± 0.084 | 19.422 ± 0.000 |
| `persistent_fused_pool2` | 21.251 ± 0.713 | 30.228 ± 1.030 | 261.867 ± 0.008 |
| `mmap_persistent_fused_pool2` | 17.326 ± 0.204 | 25.561 ± 0.056 | 2413.770 ± 0.008 |

The predeclared recommendation required both at least 10% median decode gain beyond combined MAD and median RSS no greater than 110% of baseline. `recommended_cells` is therefore empty. The mmap combination lowers the median native decode phase by about 18.58%, but sampled RSS is about 124.28× baseline, so it is explicitly **not** recommended and no default is promoted. Sampled process RSS includes file-backed mmap pages; it is not a heap-only measurement and not total system RAM. On MSVC, `clock()` measures phase-local elapsed wall time, not process CPU time or summed worker CPU time. Prefill excludes the final prompt token; decode starts by processing that token to produce the first output and then processes subsequent generated-token inputs. The immutable raw report remains at `wiki/evidence/issue-82-native-cpu-policy-report.json`; [`wiki/evidence/issue-82-timing-semantics.json`](wiki/evidence/issue-82-timing-semantics.json) is the authoritative semantic correction and links Microsoft's primary documentation.

Issue #82 is CLOSED in commit `3c8a634`; GitHub Actions run `35652844078` passed. Its negative policy result did not trigger an automatic policy bisect or default promotion.

## Native CPU capacity measurement (Issue #83)

Issue #83 adds the opt-in `--capacity-profile` payload and the caller-buffer generation API while leaving default CLI JSON unchanged and retaining the 64-position compatibility API. The new path admits up to 4096 forward positions, but the parent-verified real-model evidence exercises only 128 and 256 positions. The immutable report is `wiki/evidence/issue-83-native-capacity-report.json` (SHA-256 `4fedd8cdf44496491e92a288304c5889b9f7a4634997d8892002158fe47ef91d`).

| Actual forward positions | Prompt + generated-input positions | End-to-end wall | Sampled peak RSS |
|---:|---:|---:|---:|
| 128 | 127 + 1 | 18.159802 min | 24.699 MiB |
| 256 | 255 + 1 | 40.543901 min | 29.523 MiB |

Each row is one Windows/MSVC CPU run with a repeated-`a` prompt. Both runs emitted IDs `[1124,264]` and decoded text `" \\ a"`; that synthetic output supports no meaningful quality claim. The capacity counters establish consumed prompt/generated-input positions only; they do not prove semantic influence of the final prompt token. All frozen native executable/source inputs recorded before and after measurement are equal.

The report field `native_cpu_seconds` is again mislabeled: MSVC `clock()` is phase-local elapsed wall time, not process CPU time or summed worker CPU time. Prefill excludes the final prompt token; decode includes processing that token to produce the first output and then the generated-token input. The immutable raw report is not edited; [`wiki/evidence/issue-83-timing-semantics.json`](wiki/evidence/issue-83-timing-semantics.json) is the authoritative correction. With one run per capacity, these results do not support throughput generalization. Issue #83 is CLOSED at baseline `ec4d2fd`; GitHub Actions run `35674122057` passed.

## Native CPU 4K outcome (Issue #84)

The single native CPU 4K attempt used the #83 baseline `ec4d2fd`, 4095 prompt tokens plus one generated-token input, F32 activation, INT8 KV, and the unchanged buffered/ephemeral/baseline/serial/1-thread policies. It exited `2` after the native 28,800-second deadline was exceeded. It produced no stdout, no stderr, no report, and no completed-position count, so the 4K gate is **NOT MET** and Issue #84 remains open. This timeout does not prove that 4K capacity is impossible and does not identify CPU compute as the root cause.

The historical phase journal is failure evidence, not a valid terminal native measurement: a post-experiment exception path overwrote the native state with `phase="prepare"`, `status="failed"`, `peak_rss_bytes=0`, and `elapsed_seconds=28809.359`. Consequently final RSS is unknown (`null`). A parent observation at about 7 h 40 min recorded 142.48 MiB, but that is explicitly non-final and is not substituted for final peak RSS. The raw preparation, journal, prompt, stdout, and stderr files remain immutable and are hash-bound by [`wiki/evidence/issue-84-cpu4k-outcome.json`](wiki/evidence/issue-84-cpu4k-outcome.json).

The runner used by the attempt had SHA-256 `116212700099472958e0cbacc7cd433de0cdf93e7e16bf37dabb6b6e8fd25150`. A later, future-only journal fix retains an already-written terminal native journal; the current runner SHA-256 is `4bba43948f03d79b51b8e4a0503a84c12f796709de2f4d322bf3f8d6b2c12f6b`, and its parent verification was 60 tests passing in 3.81 s. The fix does not rewrite or reinterpret the original preparation/journal records.

Roadmap proposal: implement an opt-in CUDA path for the final output head next, with CPU-only behavior and all current CPU policies remaining the defaults. CPU 4K remains unproven; native resume, KV-quality validation, and soak/thermal validation remain deferred and must not be inferred from admission limits or the timed-out attempt.

## Opt-in CUDA final output head (Issue #85)

Issue #85 has meaningful opt-in implementation progress: the CPU-only build still links the unavailable stub, while `build_cuda_msvc.bat cuda` builds a CUDA-enabled executable and runtime selection requires explicit `--cuda-policy final-head-f32`. From the repository root, point `QX_CUDA_ROOT` (preferred) or `CUDA_PATH` at a CUDA toolkit directory, then run `build_cuda_msvc.bat cuda`; the fault runner also accepts an explicit portable override via `python tests/run_cuda_final_head_faults.py --cuda-root <cuda-toolkit-root>`. Do not record a private machine's actual toolkit path. The fixed-v2 GPU driver returned `[1124,77]`, retained legacy profile-NULL output `1124`, recorded one weight upload, two kernel launches and zero CPU fallbacks on an NVIDIA GeForce RTX 4070 Laptop GPU (compute capability 8.9). The raw run is preserved through byte-exact versioned copies and `wiki/evidence/issue-85-artifact-manifest.json`; raw-byte and LF-normalized hashes are deliberately separate.

The latest current-source strict actual-model run is `build/issue85-acceptance/actual-model-20260925T192200-914a6125/acceptance-report.json`, SHA-256 `0cba74521a13cabf3c1003385e5865a413784c813f59e804c92f186b25f98d7c`. The final-provenance manifest SHA-256 is `e572a629104dc6170e86474cd77db87ebf7a2a7de641b50fa88f775b3d1237f9`; it binds the fresh driver to production CUDA source SHA-256 `a7a43140358be5cc4950f5f0d670e303fb2ca422a383584e71e743eb2dfe5e59`, QXF SHA-256 `5609589e45a610bee6699f336109f3231326850d8f1ca839c614667c2f439840`, GGUF SHA-256 `c8c2dc330dd1ec0c72c31b12e318647e6f9e0c773b9123eccfc3d12d9acc6652`, and complete build stdout/stderr/exit-code records. For one fixed layer-47 residual, all 151936 CPU/GPU logits are finite and exact (`max_abs=0`, `RMSE=0`, cosine `1`) under strict thresholds `0.001`, `0.0001`, and `0.999999`; argmax is `1124`, and 6/6 GPU outputs are byte-identical. This is exact same-input final-head evidence, not global model parity.

The local Issue #85 publication package is **PASS**, but Issue #85 release gates are **NOT PASSED**: the exact staged digest still requires two independent reviews, followed by the Auto Research and repository CI gates on the final revision. Current contract gates pass the one-device fixed-v2 preflight, 10 fail-closed negative tests, repeated same-process bounded-memory coverage, and overlapping regression suites (`780 passed, 3 skipped`; `76 passed`; `36 passed`; `61 passed`, not summed). The negative evidence injects CUDA API return failures, including `cudaGetLastError` after a real launch; it does not demonstrate an actually failed kernel or device-loss recovery. Performance breakdown/latency acceptance is explicitly nongating. Broader device coverage, actual-kernel/device-loss testing, and performance characterization remain limitations or future work, not Issue #85 release blockers; no speed or device-loss recovery claim is authorized. CPU 4K remains a separate unmet Issue #84 gate. CPU/F32 and `--cuda-policy none` remain defaults; no default promotion, 4K claim, or release-readiness claim is authorized. Portable details and claim limits are in `wiki/evidence/issue-85-cuda-final-head-report.json`.

## Honest performance state

Earlier probe measurements on the scalar CPU path:

```text
real layer-0 probe median: ~0.2085 s/layer
real one-token 48-layer state probe: ~8.50 s
real one-token 48-layer + complete output head probe: ~8.35 s warm run
```

The complete-head measurement includes final RMSNorm and all 151936 logits for one position. Issues #82 and #83 supersede the statement that multi-token native execution was unmeasured, but their bounded cases are still not sustained conversational throughput and must not be reported as tok/s. See [`wiki/concepts/performance-model.md`](wiki/concepts/performance-model.md).

CUDA is implemented only for the opt-in F32 final output head. It is not the default and is not release-complete; attention and MoE remain on CPU, and the Issue #85 release gates above are not passed.

Long-context experimentation is opt-in and gated. The default remains
`--long-context-policy none`; `--long-context-policy ctx4k-smoke` is only an
admission/provenance gate and requires `--ctx >= 4096`. It records
`target_ctx_tokens=4096`. `--long-context-rss-limit-bytes N` is also opt-in and
defaults to `0` (disabled); the Python benchmark harness fails closed when its
sampled `peak_rss_bytes` exceeds `N`. `--long-context-kv-quality-checks` remains
`0` only and non-zero values fail closed until a real KV-quality sweep exists.
`--long-context-soak-seconds` also remains `0` only and non-zero values fail
closed until a real soak runner exists. This is not an OS hard limit and does
not run KV-quality checks, run soak tests, promote defaults or claim
speed/quality. Benchmark compact-run records preserve the validated
`long_context_profile` for each measured run so evidence keeps the active
contract attached to the timings; per-cell summaries preserve it only when all
measured runs agree, and the top-level report preserves it only when all cells
agree. The report also records a `long_context_measurement` gate with measured
ctx, cell count, run count and RSS-summary presence; this is measurement
metadata, not a throughput or stability claim.
The same metadata records `rss_limit_bytes` and whether that sampled RSS gate
was active; negative report-level RSS limits fail closed, and this does not
install an OS-level hard limit.
The inactive `none` policy also rejects non-zero report-level RSS limits before
an active sampled-RSS gate can be reported.
Report-level aggregation rejects benchmark profiles whose `enabled` flag is not
exactly `true` before deriving common profile or measurement metadata.
It also validates every profile's policy-specific `disabled_reason`: `none`
requires `none_policy`, while `ctx4k-smoke` requires null.
Missing or unsupported profile policies fail closed in the same per-cell
validation seam before profile equality or measurement derivation.
That seam also requires exact non-negative integers for all long-context
numeric fields, preventing boolean-vs-zero equality bypasses across cells.
It enforces policy-specific targets per cell too: `none` requires `0`, while
`ctx4k-smoke` requires `4096`, before profile equality or measurement.
Inactive `none` profiles also require `rss_limit_bytes=0` per cell; opt-in
`ctx4k-smoke` profiles may retain a non-negative measurement threshold.
All report profiles require `kv_quality_checks=0` per cell until a real quality
sweep exists, so unsupported provenance cannot hide behind profile drift.
They also require `soak_seconds=0` per cell until a real soak runner exists.
Report-level long-context measurement additionally requires `ctx` to be an
exact positive integer before reading cells or deriving measurement metadata.
Its `cells` container must be an exact list before emptiness or cell-shape
validation, so tuples, mappings, strings, and null fail with one contract error.
The shared profile aggregator enforces the same exact-list boundary even when
called independently of the measurement gate.
Measured run aggregation rejects an empty run set explicitly before reading the
first run's long-context profile.
It also requires the run container itself to be an exact list before emptiness
or indexing, rejecting null, tuples, mappings, and strings consistently.
Every first and later measured run must also be an object before profile
extraction or equality checks.
Every measured run must contain all required positive and non-negative metric
fields before aggregation, replacing raw missing-key failures with a contract error.
Those fields must be native integers or floats, excluding booleans and numeric
strings before normalization; finiteness and sign remain enforced by summaries.
Each run's long-context profile is validated against the report-level contract
before profiles are compared, so invalid metadata cannot masquerade as drift.
The profile field itself is required explicitly in every first and later run,
distinguishing missing metadata from null or non-object profile values.
RSS summaries with non-positive sample counts also fail closed before reporting
measured runs.
RSS summaries that omit the sample count fail closed with an explicit validation
error too.
RSS summaries must also be objects before count validation runs.
Empty benchmark reports fail closed before `long_context_measurement` derives
cell or run counts.
Benchmark cells must be objects before profile or measurement metadata is read.
Benchmark cells must also include `summary` explicitly before profile or
measurement metadata is derived.
Benchmark cells must also include `summary.long_context_profile` explicitly
before profile or measurement metadata is derived.
Inactive future counters remain fail-closed at report level too: non-zero
KV-quality checks or soak seconds are rejected until real sweeps/runners exist.

## Repository map

```text
include/                 public C headers
src/                     QXF/GGUF/runtime implementation
scripts/                 model metadata, conversion and verification helpers
tests/                   synthetic and real-model numerical gates
docs/                    chronological bootstrap evidence
wiki/                    Obsidian/Karpathy-style project knowledge base
smoke_check.py            deterministic project smoke gate
```

Open `wiki/` as an Obsidian vault. Start at [`wiki/index.md`](wiki/index.md).

## Build

### Windows / MSVC

```bat
build_msvc.bat
```

### Make-compatible C toolchain

```bash
make
```

## Tests

```bash
python -m pytest tests -q
python smoke_check.py
```

Tests requiring the real 10+ GB model skip when local model files are absent. Synthetic fixtures remain in the repository.

An optional external decoder gate links a small test helper against a local llama.cpp build. Build it only when that checkout is available:

```bat
tests\build_ggml_reference.bat
tests\build_llama_reference_oracle.bat
```

The build creates standalone residual/logit and sequence llama.cpp oracles; neither is linked into the QX runtime. The residual oracle writes selected layer inputs, internal attention/MoE checkpoints and final logits as lossless F32 sidecars. The sequence oracle keeps a persistent context for token-ID prompts and multiple greedy steps. QX can write matching sidecars with `state-loop-probe --full-moe --dump-residuals <existing-dir>` and can replay a layer suffix with `--start-layer N --residual-in layer-N.f32`. `scripts/compare_residuals.py` compares accumulated checkpoints; `scripts/compare_hybrid_residual_replay.py` separates incoming accumulated error from suffix error with exact size/finite/metadata gates; `scripts/scaled_residual_replay.py` prepares, runs and revalidates one scaled suffix cell; `scripts/scaled_residual_matrix.py` runs and revalidates the token/activation/KV matrix while separating router order from top-k membership; `scripts/compare_layer_sensitivity.py` supports perturbation analysis and same-input chains; `scripts/compare_logits.py` compares the complete vocabulary and both argmax values. Local model paths and sidecars must remain outside Git.

Current fixed-token result against llama.cpp commit `768d2a481a99cb75ec9a03b95dadbd35e7acf496`: exact end-to-end numerical parity remains **refuted**. The backward same-input sweep closes every block materially, and the modal-equivalent F16 hybrid replay attributes the remaining trajectory to accumulated/global error plus strong downstream amplification of a microscopic layer-0 discrepancy. The scaled follow-up refutes a smooth scalar response: exact-direction scales `-1` and `+1` produce final deltas of `5.73359e-4` and `4.46778` from equal input L2. The 18-cell token/activation/KV extension shows that this response is token- and modality-dependent: all three `q8_k_compat` + F16 cells retain top-8 membership, while membership transitions appear in 14/18 cells across the broader F32/F32-KV/INT8 controls. No numerical fix is authorized. The opt-in final head uses `Q6_K × Q8_K`; F32 remains the default dequantized runtime path. See [`wiki/comparisons/scaled-residual-token-modality-matrix.md`](wiki/comparisons/scaled-residual-token-modality-matrix.md), [`wiki/comparisons/scaled-layer1-residual-sensitivity.md`](wiki/comparisons/scaled-layer1-residual-sensitivity.md), and [`wiki/comparisons/hybrid-residual-replay-accumulation.md`](wiki/comparisons/hybrid-residual-replay-accumulation.md).

The final pre-head residual is explicitly compared, not inferred. Post-Q6_K Q8_K/F32-KV gives max-abs `4.68347`, RMSE `0.172092`, cosine `0.999995600`. This is worse than the prior Q5_K-only global baseline despite the layer-46 same-input fix, and is documented as quantized-path sensitivity rather than claimed as a global improvement.

Two-token sequence parity is now GREEN for the fixed matrix. For token-ID prompt `[42]`, llama F16/Q8_0 and QX F32/INT8 generate `[1124, 50853]`. For `Hello!` (`[9707, 0]`), QX generates `[358, 1184]`, matching llama F16; llama Q8_0 remains `[358, 614]`. This is not exhaustive prompt or exact-logit parity.

## Model setup

Model weights are intentionally excluded from Git. Do not commit GGUF, QXF or provider credentials.

```bash
bash scripts/download_qwen30b_iq2m.sh
build/qxqxf.exe create-from-gguf-copy   --in models/Qwen3-30B-A3B-UD-IQ2_M.gguf   --model qwen3-30b-a3b   --quant q2   --out models/Qwen3-30B-A3B-UD-IQ2_M.qxf
python scripts/export_qwen3_tokenizer.py --gguf models/Qwen3-30B-A3B-UD-IQ2_M.gguf --out models/Qwen3-30B-A3B.qxt
build/qxqxf.exe tokenizer-encode --tokenizer models/Qwen3-30B-A3B.qxt --text-file prompt.txt
```

Verify licenses and the source model card before redistributing model artifacts. This repository distributes code and documentation, not weights.

## Research method

The project uses a Karpathy-inspired Auto Research loop:

```text
question → falsifiable hypothesis → RED/baseline → minimal change
→ real execution → compare → keep/revert → document
```

See [`wiki/concepts/auto-research-loop.md`](wiki/concepts/auto-research-loop.md).

## Roadmap

1. Keep Issue #85 opt-in; freeze the exact staged digest, obtain two independent reviews, then run the Auto Research and repository CI gates on the final revision.
2. Treat broader device coverage, actual device-kernel/device-loss testing, and performance breakdown/latency measurement as nongating limitations or future work. Current evidence authorizes no speed, sustained-throughput, general leak-absence, or device-loss recovery claim.
3. Preserve CPU/F32 and `--cuda-policy none` defaults while the true Issue #85 release gates remain **NOT PASSED**.
4. Convert the existing 4K/RSS/KV-quality/soak contracts into real measurements outside heavy default CI; Issue #84 remains open and its timed-out attempt does not establish 4K capacity.

## License

MIT for repository code and documentation. Third-party model weights and external reference implementations retain their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
