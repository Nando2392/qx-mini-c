---
title: F32 vs Q8_K Activation
created: 2026-08-17
updated: 2026-08-17
type: comparison
tags: [quantization, validation, performance, cpu, cuda]
sources: [comparisons/llama-cpp-parity.md, raw/project/project-state-2026-08-17.md]
confidence: high
---

# F32 vs Q8_K activation

## Decisión

**Implementado y medido:** F32 permanece como contrato y modo predeterminado de QX. `q8_k_compat` queda disponible como modo explícito de compatibilidad/diagnóstico CPU.

`q8_k_compat` cuantiza activaciones temporales F32 a bloques Q8_K únicamente para proyecciones IQ4_XS. Los tensores Q5_K y Q6_K conservan el dot F32 existente y el runtime lo declara como `iq4_xs_q8_k_with_f32_fallback`. No existe fallback silencioso ni metadata que afirme una ruta distinta de la ejecutada.

La decisión no afirma paridad con llama.cpp. Q8_K corrige casi exactamente la primera diferencia en `Vcur`, pero la divergencia reaparece en la salida MoE y continúa hasta logits y secuencia greedy. Véanse [[llama-cpp-parity]] y [[numerical-correctness]].

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
- Metadata por ejecución distingue `not_used`, `iq4_xs_q8_k`, kernel mixto y fallback F32; workspace es cero cuando Q8_K no se usa.

El modelo real tiene mezcla de tipos: Q/K/V de capas 1, 2, 46 y 47 son Q5_K; varias proyecciones output son Q6_K. Añadir `GGML_TYPE_Q8_K` al contenedor QXF no es necesario: Q8_K es un temporal de activación, no un tensor persistido.

## Comandos reproducibles

```bash
cmd.exe /c build_msvc.bat
python -m pip install -r scripts/requirements-q8k-perf.txt
python scripts/q8k_activation_spike.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --oracle-vcur <oracle>/Vcur-0.f32 --token 42 --repetitions 5
python scripts/q8k_e2e_experiment.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --oracle <oracle> --out <temp-output>
python scripts/q8k_greedy_experiment.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --tokenizer models/Qwen3-30B-A3B.qxt
python scripts/q8k_perf_experiment.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --kv int8 --repetitions 5
python -m pytest tests/test_q8k_activation.py -q
```

Los modelos, sidecars y outputs temporales permanecen fuera de Git.

## Fidelidad numérica

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
| ffn_moe_out-0 | Q8_K compat | 0.00792122 | 0.000694041 | 0.999947515 | 1992 |
| l_out-0 | F32 | 0.00589061 | 0.000703227 | 0.999961082 | 1992 |
| l_out-0 | Q8_K compat | 0.00797009 | 0.000694425 | 0.999965249 | 1992 |
| l_out-47 | F32 | 1099.6502 | 32.9520 | 0.439279 | 940 |
| l_out-47 | Q8_K compat | 1097.6844 | 32.9117 | 0.441582 | 940 |
| final norm | F32 | 8.74427 | 1.07487 | 0.808867 | 475 |
| final norm | Q8_K compat | 8.71117 | 1.07502 | 0.809426 | 475 |
| logits | F32 | 9.093953 | 1.478256 | 0.874880 | 87787 |
| logits | Q8_K compat | 9.093479 | 1.474650 | 0.875527 | 87787 |

**Primera divergencia posterior significativa:** después de corregir atención, la diferencia reaparece en `ffn_moe_out-0`. El próximo bisect debe investigar activaciones y kernels de expertos; no debe atribuirse al KV.

Resumen de checkpoints finales para las cuatro combinaciones QX contra llama F16:

| Activación | KV | l_out-47 max_abs / cosine | final norm max_abs / cosine | logits RMSE / cosine |
|---|---|---|---|---|
| F32 | F32 | 1099.6502 / 0.439279 | 8.74427 / 0.808867 | 1.478256 / 0.874880 |
| F32 | INT8 | 1098.1270 / 0.441263 | 8.72112 / 0.809212 | 1.475123 / 0.875443 |
| Q8_K compat | F32 | 1097.6844 / 0.441582 | 8.71117 / 0.809426 | 1.474650 / 0.875527 |
| Q8_K compat | INT8 | 1096.7478 / 0.443029 | 8.69585 / 0.809633 | 1.471953 / 0.876012 |

## Greedy

| Activación | KV | `[42]` | `Hello!` |
|---|---|---|---|
| F32 | F32 | `[1124, 11287]` | `[50865, 31518]` |
| F32 | INT8 | `[1124, 11287]` | `[81379, 44707]` |
| Q8_K compat | F32 | `[1124, 11287]` | `[50865, 115810]` |
| Q8_K compat | INT8 | `[1124, 19748]` | `[53934, 256]` |
| llama F16 | — | `[1124, 50853]` | `[358, 1184]` |
| llama Q8_0 | — | `[1124, 50853]` | `[358, 614]` |

Compartir el primer argmax `1124` no implica paridad. Ningún modo QX coincide en secuencia multi-token.

## Rendimiento

Un token, 48 capas, KV INT8. Cinco repeticiones warm; no es tok/s ni throughput sostenido:

| Activación | cold | median warm | MAD | min–max | peak RSS |
|---|---:|---:|---:|---:|---:|
| F32 | 8.19598 s | 8.22912 s | 0.04541 s | 8.16235–8.27453 s | 5,603,328 B |
| Q8_K compat | 7.57912 s | 7.62247 s | 0.02807 s | 7.52635–7.79626 s | 5,603,328 B |

**Medido:** Q8_K fue aproximadamente 7.4% más rápido en mediana en este probe escalar y no cambió el peak RSS observable. El workspace adicional declarado es 4672 bytes. Hace falta perf por proyección/layer y profiling antes de generalizar.

## Trade-offs y riesgos

- **Correctitud:** mejora radicalmente atención temprana, pero no cierra MoE/logits/greedy.
- **Memoria:** 4672 bytes de workspace reutilizable; impacto despreciable frente al modelo.
- **CPU:** mejora medida, todavía sin SIMD ni thread pool.
- **Complejidad:** añade un temporal y un kernel estrecho; se limita a IQ4_XS.
- **Tipos mixtos:** Q5_K/Q6_K siguen F32 y se declaran como fallback.
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
