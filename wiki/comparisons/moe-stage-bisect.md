---
title: Qwen3MoE Stage Bisect
created: 2026-08-18
updated: 2026-08-18
type: comparison
tags: [qwen3-moe, oracle, q8-k, validation, performance]
sources: [comparisons/f32-vs-q8k-activation.md, concepts/moe-forward.md]
confidence: high
---

# Qwen3MoE stage bisect

## Resultado

El bisect aislado de layer 0 usa exactamente el mismo `ffn_inp-0.f32` en QX y llama.cpp. Identificó que el router, top-8, renormalización, SiLU, mezcla y layouts QX son correctos. La diferencia de `ffn_moe_out-0` procedía principalmente del contrato de activación de los expertos CPU: ggml ejecuta `IQ2_XS × Q8_K` y `IQ3_XXS × Q8_K`, mientras QX usaba pesos decodificados contra F32.

`q8_k_compat` se amplió, sólo de forma opt-in, a gate/up IQ2_XS y down IQ3_XXS. F32 continúa siendo el default. En este milestone, tipos distintos mantenían fallback F32 explícito; [[iq2-s-iq4-xs-q8k]] y [[layer1-layer2-sensitivity]] validaron después tipos 22/23 y Q5_K.

Con el `ffn_inp` exacto del oracle, todas las etapas de layer 0 quedan dentro de max_abs `1.20e-6`. En el forward end-to-end, `ffn_moe_out-0` mejora de RMSE `6.77590e-4` a `1.40910e-4` y `l_out-0` de `7.03227e-4` a `1.41082e-4` frente a llama F16.

En este baseline, la primera divergencia material pasó a **layer 2** y layer 1 `(22,22,23)` conservaba fallback F32. Los goldens posteriores validaron esos tipos y el fix Q5_K redujo layer-2 RMSE a `0.00660423`; véase [[layer1-layer2-sensitivity]].

## Fuente primaria fijada

Oracle read-only:

```text
llama.cpp commit 768d2a481a99cb75ec9a03b95dadbd35e7acf496
```

| Fuente | Líneas/símbolos | Contrato observado |
|---|---|---|
| `src/models/qwen3moe.cpp` | `ffn_inp`, `ffn_norm`, `build_moe_ffn`, `ffn_moe_out`, `l_out` | residual attention → RMSNorm → MoE → suma residual |
| `src/llama-graph.cpp` | `ffn_moe_probs`, `ffn_moe_topk` | softmax sobre 128 expertos y top-8 |
| `src/llama-graph.cpp` | `ffn_moe_weights_sum`, `ffn_moe_weights_norm` | renormalización de los ocho pesos seleccionados |
| `src/llama-graph.cpp` | `ffn_moe_gate`, `ffn_moe_up`, `ffn_moe_swiglu` | proyecciones separadas y `SiLU(gate) × up` |
| `src/llama-graph.cpp` | `ffn_moe_down`, `ffn_moe_weighted`, `ffn_moe_out` | down por experto, peso normalizado y suma de ocho contribuciones |
| `ggml/src/ggml-cpu/arch/*/quants.c` | vec-dot IQ2_XS/IQ3_XXS con Q8_K | temporal Q8_K CPU para expertos compatibles |

Los callbacks son capturables con `cb_eval`; el oracle exporta sidecars F32 lossless y registra commit/FNV-1a64. llama.cpp no forma parte del runtime QX.

## Contrato del probe

```bash
build/qxqxf.exe moe-stage-probe \
  --in models/Qwen3-30B-A3B-UD-IQ2_M.qxf \
  --layer 0 \
  --ffn-inp <oracle>/ffn_inp-0.f32 \
  --out-dir <temp>/qx-moe-stage \
  --activation q8_k_compat
python scripts/compare_moe_stages.py --qx-dir <temp>/qx-moe-stage --llama-dir <oracle> --layer 0
```

Exporta 12 sidecars: norm, logits, probs, top-k, pesos seleccionados, suma, pesos normalizados, gate, up, SwiGLU, down y contribuciones ponderadas. El `ffn_moe_out` se reconstruye sumando las ocho contribuciones. El probe falla cerrado ante tamaño corto/largo, NaN/Inf, capa inválida, output no escribible, overflow y layout incompatible.

El oracle también captura `ffn_moe_out-0` y `l_out-0`; el test no reutiliza resultados QX como expected.

## Matriz por etapa, mismo `ffn_inp`

Token 42, layer 0, llama F16, activación Q8_K compatible:

| Etapa | count | max_abs | RMSE | cosine | índice max |
|---|---:|---:|---:|---:|---:|
| ffn_norm | 2048 | 2.38419e-7 | 1.71447e-8 | 1.000000000000001 | 571 |
| router logits | 128 | 1.90735e-6 | 5.94274e-7 | 0.999999999999999 | 112 |
| router probs | 128 | 2.23517e-8 | 2.94366e-9 | 0.999999999999984 | 89 |
| top-8 ids | 8 | 0 | 0 | 1 | 0 |
| selected weights | 8 | 2.23517e-8 | 1.00307e-8 | 0.999999999999990 | 1 |
| weights sum | 1 | 2.98023e-8 | 2.98023e-8 | 1 | 0 |
| weights normalized | 8 | 4.47035e-8 | 2.07415e-8 | 0.999999999999989 | 1 |
| gate | 6144 | 7.15256e-7 | 5.48163e-8 | 0.999999999999992 | 5096 |
| up | 6144 | 3.57628e-7 | 3.60751e-8 | 0.999999999999991 | 2435 |
| SwiGLU | 6144 | 7.15256e-7 | 2.43537e-8 | 0.999999999999990 | 2435 |
| down | 16384 | 1.19209e-6 | 2.65987e-8 | 0.999999999999975 | 8136 |
| weighted | 16384 | 2.08616e-7 | 4.91117e-9 | 0.999999999999988 | 4040 |

Top-8 exacto: `[49, 89, 92, 48, 108, 58, 4, 38]`. Pesos normalizados QX: `[0.211989388, 0.206715435, 0.124410488, 0.109741203, 0.108423740, 0.0886193812, 0.0847552717, 0.0653451085]`; suma dentro del redondeo F32 de `1.0`.

### Máximo por rank de experto

| rank / expert | gate | up | SwiGLU | down | weighted |
|---|---:|---:|---:|---:|---:|
| 0 / 49 | 2.38419e-7 | 1.78814e-7 | 4.76837e-7 | 7.15256e-7 | 1.78814e-7 |
| 1 / 89 | 2.38419e-7 | 1.78814e-7 | 5.96046e-7 | 4.76837e-7 | 2.08616e-7 |
| 2 / 92 | 2.38419e-7 | 2.38419e-7 | 5.96046e-7 | 1.49012e-7 | 1.86265e-8 |
| 3 / 48 | 2.38419e-7 | 3.57628e-7 | 7.15256e-7 | 1.19209e-6 | 1.19209e-7 |
| 4 / 108 | 1.19209e-7 | 1.19209e-7 | 1.78814e-7 | 1.49012e-7 | 1.86265e-8 |
| 5 / 58 | 2.38419e-7 | 1.78814e-7 | 3.57628e-7 | 3.57628e-7 | 1.49012e-8 |
| 6 / 4 | 7.15256e-7 | 2.38419e-7 | 2.38419e-7 | 1.78814e-7 | 1.49012e-8 |
| 7 / 38 | 2.38419e-7 | 1.78814e-7 | 2.38419e-7 | 1.19209e-7 | 3.72529e-9 |

La primera desviación de experto medible está en gate, rank 6/expert 4 (`7.15256e-7`); es ruido F32, no una divergencia material. El máximo global aparece después en down, rank 3/expert 48 (`1.19209e-6`).

## Tipos y strides reales

| layer | gate/up/down ggml_type | row bytes gate/up/down |
|---:|---|---|
| 0 | 17 / 17 / 18 | 592 / 592 / 294 |
| 1 | 22 / 22 / 23 | 656 / 656 / 408 |
| 24 | 17 / 17 / 21 | 592 / 592 / 330 |
| 47 | 22 / 22 / 23 | 656 / 656 / 408 |

No se asume layout homogéneo. El modo Q8_K usa actualmente tipos 17/18 en MoE. Los demás tipos caen a F32 y se declaran como `iq2_xs_iq3_xxs_q8_k_with_f32_fallback` cuando la ejecución mezcla ambos caminos.

## Primera divergencia por capa

Comparación de inputs de las 48 capas, QX Q8_K/KV F32 contra llama F16:

| layer | max_abs | RMSE | cosine | gate material |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 1 | PASS |
| 1 | 0.000750065 | 0.000141082 | 0.999998410 | PASS |
| 2 | 27.818619 | 0.617845 | 0.999687810 | **FAIL: primera material** |
| 24 | 45.042236 | 1.000039 | 0.999994460 | FAIL |
| 47 | 69.561829 | 1.897163 | 0.998256625 | FAIL |

Gate material usado: max_abs ≤`1e-3` y cosine ≥`0.99999`. El gate estricto `1e-5` marca layer 1, como corresponde al ruido cuantizado residual.

## Baseline histórico pre-Q5_K: E2E, logits y greedy

| Activación / KV | ffn_moe_out-0 RMSE / cosine | l_out-0 RMSE / cosine | final norm RMSE / cosine | logits RMSE / cosine |
|---|---|---|---|---|
| F32 / F32 | 6.77590e-4 / 0.999944290 | 7.03227e-4 / 0.999961082 | 1.074871 / 0.808867 | 1.478256 / 0.874880 |
| Q8_K / F32 | **1.40910e-4 / 0.999997595** | **1.41082e-4 / 0.999998410** | 1.074959 / 0.810037 | 1.472742 / 0.875870 |
| Q8_K / INT8 | 4.91946e-4 / 0.999971785 | 5.10055e-4 / 0.999979364 | 1.074976 / 0.810359 | 1.469973 / 0.876367 |

Todos conservaban argmax `1124`, pero en este baseline no había paridad greedy:

| Activación / KV | `[42]` | `Hello!` |
|---|---|---|
| F32 / F32 | `[1124, 11287]` | `[50865, 31518]` |
| Q8_K / F32 | `[1124, 11287]` | `[50865, 31518]` |
| Q8_K / INT8 | `[1124, 11287]` | `[50865, 28065]` |
| llama F16 | `[1124, 50853]` | `[358, 1184]` |

Post-Q5_K, QX F32/INT8 coincide con esa fila llama F16 en la matriz fija. La tabla se conserva como evidencia causal del milestone MoE.

## Rendimiento

Un token, 48 capas, KV INT8, cinco repeticiones warm:

| Activación | cold | median warm | MAD | min–max | median peak RSS |
|---|---:|---:|---:|---:|---:|
| F32 | 11.87577 s | 9.24408 s/token | 0.46075 s | 8.78333–13.96411 s | 5,672,960 B |
| Q8_K compat | 4.25738 s | 3.88918 s/token | 0.04593 s | 3.80302–4.32378 s | 5,603,328 B |

Medición observada: Q8_K fue `2.37687×` más rápido, reducción de latencia mediana `57.9279%`; RSS mediana `1.2274%` menor. El baseline F32 tuvo variabilidad alta, por lo que estos valores describen este probe y máquina, no throughput sostenido. Workspace Q8_K declarado: 4672 B para las proyecciones attention; máximo MoE por activación hidden: 2336 B.

## Reproducción y gates

```bash
cmd.exe /c build_msvc.bat
cmd.exe /c tests/build_llama_reference_oracle.bat
cmd.exe /c tests/build_ggml_reference.bat
python -m pytest tests/test_moe_stage_oracle.py -q
python -m pytest tests/test_compare_moe_stages.py -q
python -m pytest tests/test_q8k_activation.py tests/test_real_iq4xs_projection.py -q
python scripts/q8k_e2e_experiment.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --oracle <oracle> --out <temp>
python scripts/q8k_greedy_experiment.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --tokenizer models/Qwen3-30B-A3B.qxt
python scripts/q8k_perf_experiment.py --qx-exe build/qxqxf.exe --source-model models/Qwen3-30B-A3B-UD-IQ2_M.gguf --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --tokenizer models/Qwen3-30B-A3B.qxt --prompt-file tests/fixtures/q8k_perf_prompt.txt --output <report.json> --kv int8 --repetitions 5
```

Modelos, tokenizers, ejecutables, sidecars y JSON de resultados permanecen fuera de Git.

Relaciones: [[moe-forward]], [[f32-vs-q8k-activation]], [[numerical-correctness]], [[current-status-and-roadmap]].
