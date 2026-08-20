---
title: F32 vs Q8_K Activation
created: 2026-08-17
updated: 2026-08-18
type: comparison
tags: [quantization, validation, performance, cpu, cuda]
sources: [comparisons/llama-cpp-parity.md, raw/project/project-state-2026-08-17.md]
confidence: high
---

# F32 vs Q8_K activation

## Decisión

**Implementado y medido:** F32 permanece como contrato y modo predeterminado de QX. `q8_k_compat` queda disponible como modo explícito de compatibilidad/diagnóstico CPU.

`q8_k_compat` cuantiza activaciones temporales F32 a bloques Q8_K para proyecciones IQ4_XS y, tras [[moe-stage-bisect]] y [[iq2-s-iq4-xs-q8k]], para expertos IQ2_XS/IQ3_XXS/IQ2_S/IQ4_XS. Los demás tipos conservan dot F32 y el runtime declara si ejecutó ruta Q8_K, F32 o mixta. No existe fallback silencioso ni metadata que afirme una ruta distinta de la ejecutada.

La decisión no afirma paridad numérica exacta con llama.cpp. Las tablas pre-Q5_K de esta ADR mostraban divergencia desde layer 1 hasta greedy; [[layer1-layer2-sensitivity]] las supersede: corrige Q5_K, reduce logits a RMSE `0.0346769` y pasa la matriz greedy fija de dos prompts/dos tokens contra llama F16. Véase [[numerical-correctness]] para el estado vivo.

## Hipótesis

1. La primera diferencia `Vcur` procede de `IQ4_XS × F32` en QX frente a `IQ4_XS × Q8_K` en ggml CPU.
2. Reproducir el temporal Q8_K debe acercar `Vcur` y checkpoints de atención al oracle.
3. Esa corrección puede no cerrar la divergencia end-to-end si los kernels MoE mantienen otros contratos numéricos.
4. El modo no debe convertirse en default sin evidencia de logits, greedy, coste y compatibilidad futura.

## Fuente primaria y contrato

Oracle fijado a commit llama.cpp:

```text
768d2a481a99cb75ec9a03b95dadbd35e7acf496
```

Fuentes inspeccionadas:

| Archivo | Función/contrato | Implicación QX |
|---|---|---|
| `ggml/src/ggml-quants.c` | `quantize_row_q8_K_ref` | bloque de 256, escala F32, 256 `int8`, 16 sumas `int16` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | traits IQ4_XS | activación vec-dot registrada como Q8_K |
| `ggml/src/ggml-cpu/arch/*/quants.c` | `ggml_vec_dot_iq4_xs_q8_K` | producto y acumulación CPU de referencia |
| `ggml/src/ggml-cuda/*` | traits/dequant IQ4_XS | CUDA usa rutas propias; no demuestra temporal Q8_K equivalente |

Layout validado:

```text
QK_K = 256
float d                  4 bytes
int8 qs[256]           256 bytes
int16 bsums[16]         32 bytes
block_q8_K total       292 bytes
```

QX y llama.cpp están bajo licencia MIT. La implementación QX adapta el contrato matemático estrecho; no enlaza ni envuelve llama.cpp como runtime. llama.cpp permanece oracle read-only.

## Implementación

**Implementado:**

- `--activation f32` continúa siendo default.
- `--activation q8_k_compat` activa el temporal Q8_K para IQ4_XS.
- Workspace fijo reutilizable: 16 bloques, 4672 bytes, suficiente para activaciones de 4096 valores.
- No hay allocation por fila, experto o capa para el temporal Q8_K.
- Bloques parciales, NaN/Inf, tamaños fuera de rango y overflow del parser fallan cerrados.
- Probe sintético: `q8-k-activation-probe`.
- Los 292 bytes completos coinciden con `quantize_row_q8_K_ref` para entradas mixed, positivas, negativas y extremos alternos ±1.
- Metadata por ejecución distingue `not_used`, `iq4_xs_q8_k`, kernel mixto y fallback F32; para MoE publica además gate/up y down por separado, por lo que una combinación heterogénea no se colapsa a una etiqueta genérica. El workspace es cero cuando Q8_K no se usa.

El modelo real tiene mezcla de tipos: Q/K/V de capas 1, 2, 46 y 47 son Q5_K; varias proyecciones output son Q6_K. Añadir `GGML_TYPE_Q8_K` al contenedor QXF no es necesario: Q8_K es un temporal de activación, no un tensor persistido.

## Comandos reproducibles

```bash
cmd.exe /c build_msvc.bat
python -m pip install -r scripts/requirements-q8k-perf.txt
python scripts/q8k_activation_spike.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --oracle-vcur <oracle>/Vcur-0.f32 --token 42 --repetitions 5
python scripts/q8k_e2e_experiment.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --oracle <oracle> --out <temp-output>
python scripts/q8k_greedy_experiment.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --tokenizer models/Qwen3-30B-A3B.qxt
python scripts/q8k_perf_experiment.py --qx-exe build/qxqxf.exe --source-model models/Qwen3-30B-A3B-UD-IQ2_M.gguf --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --tokenizer models/Qwen3-30B-A3B.qxt --prompt-file tests/fixtures/q8k_perf_prompt.txt --output <report.json> --kv int8 --repetitions 5
python -m pytest tests/test_q8k_activation.py -q
```

Los modelos, sidecars y outputs temporales permanecen fuera de Git.

## Baseline histórico pre-Q5_K: fidelidad numérica

> Estas tablas conservan la medición que motivó el bisect. No son el estado post-Q5_K; véase [[layer1-layer2-sensitivity]].

Token `[42]`, oracle llama F16, KV F32 en QX:

| Checkpoint | QX mode | max_abs | RMSE | cosine | first max index |
|---|---|---:|---:|---:|---:|
| layer-0 input | F32 | 0 | 0 | 1 | 0 |
| layer-0 input | Q8_K compat | 0 | 0 | 1 | 0 |
| Vcur-0 | F32 | 0.000305031 | 0.0000831637 | 0.999943484 | 137 |
| Vcur-0 | Q8_K compat | 7.45058e-9 | 1.15833e-9 | 0.99999999999999 | 323 |
| kqv_out-0 | F32 | 0.000305337 | 0.0000830610 | 0.999943621 | 1033 |
| kqv_out-0 | Q8_K compat | 0.0000148565 | 1.63887e-6 | 0.999999978 | 3107 |
| ffn_inp-0 | F32 | 0.00333905 | 0.000183834 | 0.999976473 | 1992 |
| ffn_inp-0 | Q8_K compat | 0.0000489354 | 2.28653e-6 | 0.999999996 | 1992 |
| ffn_moe_out-0 | F32 | 0.00255167 | 0.000677590 | 0.999944290 | 1992 |
| ffn_moe_out-0 | Q8_K compat | 0.000701189 | 0.000140910 | 0.999997595 | 1992 |
| l_out-0 | F32 | 0.00589061 | 0.000703227 | 0.999961082 | 1992 |
| l_out-0 | Q8_K compat | 0.000750065 | 0.000141082 | 0.999998410 | 1992 |
| l_out-47 | F32 | 1099.6502 | 32.9520 | 0.439279 | 940 |
| l_out-47 | Q8_K compat | 1095.7421 | 32.8627 | 0.444373 | 940 |
| final norm | F32 | 8.74427 | 1.07487 | 0.808867 | 475 |
| final norm | Q8_K compat | 8.65504 | 1.07496 | 0.810037 | 475 |
| logits | F32 | 9.093953 | 1.478256 | 0.874880 | 87787 |
| logits | Q8_K compat | 9.087952 | 1.472742 | 0.875870 | 87787 |

**Bisect posterior:** [[moe-stage-bisect]] cerró los kernels de expertos layer 0. [[iq2-s-iq4-xs-q8k]] cerró `IQ2_S/IQ4_XS × Q8_K` de layer 1 con input idéntico. La primera divergencia material sigue en input layer 2 por amplificación de la perturbación previa.

Resumen de checkpoints finales para las cuatro combinaciones QX contra llama F16:

| Activación | KV | l_out-47 max_abs / cosine | final norm max_abs / cosine | logits RMSE / cosine |
|---|---|---|---|---|
| F32 | F32 | 1099.6502 / 0.439279 | 8.74427 / 0.808867 | 1.478256 / 0.874880 |
| F32 | INT8 | 1098.1270 / 0.441263 | 8.72112 / 0.809212 | 1.475123 / 0.875443 |
| Q8_K compat | F32 | 1132.7604 / 0.411556 | 9.96629 / 0.797838 | 1.480970 / 0.874416 |
| Q8_K compat | INT8 | 1133.1087 / 0.411358 | 9.98870 / 0.797812 | 1.478728 / 0.874826 |

## Baseline histórico pre-Q5_K: greedy

| Activación | KV | `[42]` | `Hello!` |
|---|---|---|---|
| F32 | F32 | `[1124, 11287]` | `[50865, 31518]` |
| F32 | INT8 | `[1124, 11287]` | `[81379, 44707]` |
| Q8_K compat | F32 | `[1124, 11287]` | `[50865, 46709]` |
| Q8_K compat | INT8 | `[1124, 19748]` | `[50865, 118860]` |
| llama F16 | — | `[1124, 50853]` | `[358, 1184]` |
| llama Q8_0 | — | `[1124, 50853]` | `[358, 614]` |

En este baseline pre-Q5_K, compartir el primer argmax `1124` no implicaba paridad y ningún modo QX coincidía. Post-Q5_K, la matriz fija sí coincide con llama F16; no se extrapola a cobertura exhaustiva.

## Baseline histórico pre-Q5_K: rendimiento

Un token, 48 capas, KV INT8. Cinco repeticiones warm; no es tok/s ni throughput sostenido:

| Activación | cold | median warm | MAD | min–max | median peak RSS |
|---|---:|---:|---:|---:|---:|
| F32 | 8.31307 s | 8.00310 s | 0.13300 s | 7.75351–8.15110 s | 5,611,520 B median |
| Q8_K compat | 2.36520 s | 2.28223 s | 0.01248 s | 2.25328–2.30621 s | 5,615,616 B median |

**Medido tras extender MoE a IQ2_S/IQ4_XS:** Q8_K fue `3.50670×` más rápido en mediana (`71.4832%` menos latencia). El RSS mediano aumentó sólo `4096 B`, una página tratada como ruido. Son segundos/token de este probe, no throughput sostenido. Workspace adicional declarado: 4672 bytes para attention y hasta 2336 bytes por activación MoE.

Benchmark post-Q5_K actual: F32 `8.64031 s/token`, Q8_K `2.41209 s/token`, speedup `3.58208×`, RSS mediano idéntico `5,627,904 B`.

## Trade-offs y riesgos

- **Correctitud:** cierra vec-dot y etapas MoE/atención con input idéntico. La propagación/logits no son bit-exactos; la matriz greedy fija post-Q5_K sí pasa.
- **Memoria:** 4672 bytes de workspace reutilizable; impacto despreciable frente al modelo.
- **CPU:** mejora medida, todavía sin SIMD ni thread pool.
- **Complejidad:** añade un temporal y dispatchs estrechos para IQ4_XS, Q5_K, IQ2_XS, IQ3_XXS e IQ2_S; no generaliza por analogía a otros tipos.
- **Tipos mixtos:** Q6_K/IQ3_S y cualquier tipo sin golden siguen F32 y se declaran como fallback. Q5_K queda validado por [[layer1-layer2-sensitivity]].
- **CUDA:** no se promoverá este temporal a contrato CUDA sin golden y benchmark del backend real.
- **Paridad:** `q8_k_compat` significa compatibilidad de la proyección IQ4_XS CPU, no paridad global.

## Condición de revisión

Revisar esta ADR si se cumple alguna:

1. El bisect MoE implementa temporales compatibles y demuestra mejora end-to-end.
2. Q8_K logra secuencia/logits equivalentes con tolerancias publicadas.
3. Un kernel SIMD/threaded demuestra ventaja estable sin regresión numérica.
4. El backend CUDA necesita un contrato de activación distinto.
5. Benchmarks 4K/RSS/sostenidos muestran coste o inestabilidad no visible aquí.

Roadmap: [[current-status-and-roadmap]]. Rendimiento: [[performance-model]].
