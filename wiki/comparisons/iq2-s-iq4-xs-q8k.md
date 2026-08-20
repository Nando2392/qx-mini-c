---
title: IQ2_S and IQ4_XS Expert Q8_K Gate
created: 2026-08-18
updated: 2026-08-18
type: comparison
tags: [qwen3-moe, quantization, q8-k, cpu, oracle, validation]
sources: [comparisons/moe-stage-bisect.md, comparisons/f32-vs-q8k-activation.md]
confidence: high
---

# IQ2_S and IQ4_XS expert Q8_K gate

## Decisión

**Implementado y validado en CPU como modo opt-in.** `q8_k_compat` usa ahora activaciones temporales Q8_K para gate/up `IQ2_S` y down `IQ4_XS` de Qwen3MoE cuando la pareja de tipos es compatible. F32 continúa siendo el modo predeterminado. Los tipos sin golden conservan fallback F32 explícito.

El alcance corresponde al issue GitHub #10. No demuestra paridad global: con el mismo `ffn_inp-1`, las etapas MoE de layer 1 coinciden con ggml dentro de error de redondeo; end-to-end, la diferencia pequeña al entrar en layer 1 se amplifica en el experto dominante y la primera divergencia material sigue en input layer 2.

## Corrección de alcance

La fuente primaria del oracle fijado define:

```text
GGML_TYPE_IQ3_S  = 21
GGML_TYPE_IQ2_S  = 22
GGML_TYPE_IQ4_XS = 23
```

Por tanto, los tipos `(22,22,23)` observados en layers 1 y 47 son `IQ2_S/IQ2_S/IQ4_XS`, no `IQ2_S/IQ3_S`. El issue se corrigió antes de implementar.

## Fuente primaria y layout

Oracle read-only:

```text
C:/Users/fjmn2/Dev/llama.cpp-k3
commit 768d2a481a99cb75ec9a03b95dadbd35e7acf496
```

Contratos inspeccionados:

| Tipo | ggml vec-dot | Bloque peso | Row bytes real |
|---|---|---:|---:|
| IQ2_S gate/up | `ggml_vec_dot_iq2_s_q8_K` | 82 B / 256 valores | 656 B para 2048 entradas |
| IQ4_XS down | `ggml_vec_dot_iq4_xs_q8_K` | 136 B / 256 valores | 408 B para 768 entradas |
| Activación | `block_q8_K` | 292 B / 256 valores | 2336 B para 2048; 876 B para 768 |

El helper llama `ggml_cpu_init()` y obtiene `vec_dot` mediante traits públicos de `ggml-cpu.h`. QX no enlaza ni envuelve llama.cpp en runtime.

## TDD y goldens independientes

El ciclo RED→GREEN cubrió:

1. `internals=1` en el oracle para exportar 18 checkpoints de layer 1 sin cambiar el default `internals` de layer 0.
2. Filas reales `IQ2_S × Q8_K` y `IQ4_XS × Q8_K` contra el helper ggml.
3. Forward MoE completo de layer 1 con exactamente el mismo `ffn_inp-1.f32`.
4. Metadata por ruta ejecutada y agregación de familias en el state loop.
5. Checkpoint E2E explícito `layer-2-input`.

Filas verificadas por tensor:

```text
blk.1.ffn_gate_exps.weight: rows 0, 384, 767; expert 0
blk.1.ffn_down_exps.weight: rows 0, 1024, 2047; expert 0
```

## Resultado por etapa, layer 1

Input idéntico en QX/llama.cpp. Expertos seleccionados:

```text
[68, 114, 55, 90, 0, 9, 28, 73]
```

| Etapa | max_abs | RMSE | cosine | max index |
|---|---:|---:|---:|---:|
| norm | 4.76837e-7 | 1.88237e-8 | 0.999999999999997 | 571 |
| router logits | 9.53674e-7 | 4.66480e-7 | 0.999999999999998 | 4 |
| gate | 9.53674e-7 | 5.49761e-8 | 0.999999999999985 | 711 |
| up | 3.81470e-6 | 7.08511e-8 | 0.999999999999989 | 711 |
| SwiGLU | 4.57764e-5 | 5.85814e-7 | 1.000000000000002 | 711 |
| down | 1.22070e-4 | 9.55361e-7 | 0.999999999999996 | 940 |
| weighted | 4.57764e-5 | 3.57930e-7 | 0.999999999999998 | 940 |
| mezcla final | 4.34666e-5 | 9.61547e-7 | 1.000000000000001 | 940 |

`compare_moe_stages.py` devolvió `passed=true`; todas las métricas fueron finitas.

## Resultado end-to-end

Token `42`, oracle llama F16:

| Activación QX | KV | layer-1 input max/RMSE/cos | layer-2 input max/RMSE/cos | logits RMSE/cos | argmax |
|---|---|---|---|---|---:|
| F32 | F32 | 0.00589061 / 0.000703227 / 0.999961082 | 27.2744 / 0.605843 / 0.999693464 | 1.47826 / 0.874880 | 1124 |
| F32 | INT8 | 0.00626087 / 0.000742375 / 0.999957401 | 27.6382 / 0.613862 / 0.999690799 | 1.47512 / 0.875443 | 1124 |
| Q8_K | F32 | 0.000750065 / 0.000141082 / 0.999998410 | 28.2042 / 0.626792 / 0.999695728 | 1.48097 / 0.874416 | 1124 |
| Q8_K | INT8 | 0.00212979 / 0.000510055 / 0.999979364 | 27.8637 / 0.619276 / 0.999698441 | 1.47873 / 0.874826 | 1124 |

El nuevo kernel reduce el error de entrada a layer 1 frente a F32, pero no elimina la sensibilidad del MoE: el experto 68 tiene activaciones grandes y amplifica la perturbación previa. El siguiente bisect debe separar atención/residual de layer 1 y sensibilidad del experto dominante; no debe atribuir el error al vec-dot ya validado con input idéntico.

> [!NOTE]
> Esta tabla preserva el baseline del issue #10. [[layer1-layer2-sensitivity]] la supersede tras corregir Q5_K: Q8_K/F32-KV baja layer-2-input de max-abs `28.2042`/RMSE `0.626792` a `0.294106`/`0.00660423`, y logits RMSE de `1.48097` a `0.0346769`.

## Greedy

| Activación | KV | `[42]` | `Hello!` |
|---|---|---|---|
| F32 | F32 | `[1124, 11287]` | `[50865, 31518]` |
| F32 | INT8 | `[1124, 11287]` | `[81379, 44707]` |
| Q8_K | F32 | `[1124, 11287]` | `[50865, 46709]` |
| Q8_K | INT8 | `[1124, 19748]` | `[50865, 118860]` |
| llama F16 | — | `[1124, 50853]` | `[358, 1184]` |

Ningún modo QX alcanza paridad secuencial.

## Rendimiento

Un token, 48 capas, KV INT8, cinco repeticiones warm:

| Activación | median | MAD | min–max | peak RSS median |
|---|---:|---:|---:|---:|
| F32 | 8.00310 s/token | 0.13300 s | 7.75351–8.15110 s | 5,611,520 B |
| Q8_K | 2.28223 s/token | 0.01248 s | 2.25328–2.30621 s | 5,615,616 B |

Speedup observado: `3.50670×`; reducción de latencia: `71.4832%`. La diferencia RSS de `+4096 B` es una página y se trata como ruido, no como regresión material.

## Metadata y fallback

- layer 0: `iq2_xs_q8_k_and_iq3_xxs_q8_k`
- layers 1/47: `iq2_s_q8_k_and_iq4_xs_q8_k`
- layers 0–1 agregados: `iq2_xs_iq3_xxs_iq2_s_iq4_xs_q8_k`
- ruta heterogénea con tipo no soportado: `q8_k_expert_kernels_with_f32_fallback`
- F32: `dequant_f32`

El campo agregado se conserva por compatibilidad. Los campos canónicos exactos separan cada subruta:

- `gate_up_projection_kernel` / `down_projection_kernel` en `moe-stage-probe`;
- `moe_gate_up_projection_kernel` / `moe_down_projection_kernel` en el state loop;
- layers 0–1: gate/up `iq2_xs_iq2_s_q8_k`, down `iq3_xxs_iq4_xs_q8_k`;
- layer 24: gate/up `iq2_xs_q8_k`, down `dequant_f32`.

Así, una combinación mixta o un fallback en un solo rol no queda oculto por una etiqueta agregada genérica. La metadata se deriva de tipos y kernels ejecutados, no del modo solicitado.

## Comandos reproducibles

```bash
cmd.exe /c build_msvc.bat
cmd.exe /c tests/build_llama_reference_oracle.bat
cmd.exe /c tests/build_ggml_reference.bat
python -m pytest tests/test_moe_stage_oracle.py tests/test_real_iq4xs_projection.py tests/test_q8k_e2e_experiment.py -q
python scripts/compare_moe_stages.py --qx-dir <qx-layer1> --llama-dir <oracle-layer1> --layer 1
python scripts/q8k_e2e_experiment.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --oracle <oracle> --out <temp>
python scripts/q8k_greedy_experiment.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --tokenizer models/Qwen3-30B-A3B.qxt
python scripts/q8k_perf_experiment.py --qx-exe build/qxqxf.exe --source-model models/Qwen3-30B-A3B-UD-IQ2_M.gguf --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --tokenizer models/Qwen3-30B-A3B.qxt --prompt-file tests/fixtures/q8k_perf_prompt.txt --output <report.json> --kv int8 --repetitions 5
```

Modelos, sidecars, JSON de experimento, EXE/OBJ/DLL y dumps permanecen fuera de Git.

Roadmap: [[current-status-and-roadmap]]. Contrato general: [[f32-vs-q8k-activation]]. Bisect previo: [[moe-stage-bisect]].
