---
title: Persistent Scratch Buffers
created: 2026-08-21
updated: 2026-08-21
type: comparison
tags: [performance, cpu, scratch, allocations, lifecycle]
sources: [issue-25, qx_format.c, q8k_perf_experiment.py, evidence/issue-25-persistent-scratch-baseline.json]
confidence: high
---

# Persistent scratch buffers

Issue #25 evalúa la prioridad CPU 2 desde [[optimization-priorities]]: reutilizar buffers scratch dentro de una ejecución para eliminar `malloc`/`calloc`/`free` repetidos del hot path. El cambio queda opt-in mediante `--scratch-policy persistent`; `ephemeral` sigue siendo el default/control, igual que [[qxf-mmap-io]] mantiene buffered como control I/O.

## Causalidad

El inventario estático + profiling nativo localizó churn material en:

- atención RoPE/GQA por capa/token: `qbuf`, `obuf`, `qfloat`, `context`, `scores`, `weights`;
- MoE por capa/token: buffer combinado para `ffn_input`, `moe_output`, `expert_output`, `gate_values`, `up_values`, `expert_hidden`;
- spans QXF buffered para pesos/filas/expertos/final head, que mmap evita en parte pero conserva su coste de RSS.

La implementación de Issue #25 sólo reutiliza los scratch temporales de atención y MoE. No cambia formato QXF, matemáticas, kernels, threading, SIMD, CUDA, expert cache, prefill GEMM, speculative decoding ni KV2.

## Evidencia

Fuente: `wiki/evidence/issue-25-persistent-scratch-baseline.json`, schema 4, matriz 2×2×2:

- políticas scratch: `ephemeral`, `persistent`;
- activaciones: `f32`, `q8_k_compat`;
- backends I/O: `buffered`, `mmap`;
- 1 warm-up + 3 mediciones por celda;
- `--generate 1`, `--layers 48`, `--ctx 16`, `--kv int8`, `--seed 7`, `--full-moe`, `--final-head`, `--bench`.

| Política | Backend | Activación | malloc median | free median | bytes solicitados median | scratch peak | total median s | peak RSS median |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ephemeral | buffered | f32 | 49136 | 54965 | 2638435584 | 0 | 17.1548896 | 20406272 |
| persistent | buffered | f32 | 48656 | 54293 | 2630661376 | 65536 | 17.4008302 | 20406272 |
| ephemeral | buffered | q8_k_compat | 49136 | 54965 | 2638435584 | 0 | 4.3797097 | 20406272 |
| persistent | buffered | q8_k_compat | 48656 | 54293 | 2630661376 | 65536 | 4.1461890 | 20406272 |
| ephemeral | mmap | f32 | 5742 | 54965 | 1599966464 | 0 | 17.5682498 | 1964691456 |
| persistent | mmap | f32 | 5262 | 54293 | 1592192256 | 65536 | 17.1050645 | 1964744704 |
| ephemeral | mmap | q8_k_compat | 5742 | 54965 | 1599966464 | 0 | 4.6015538 | 1964699648 |
| persistent | mmap | q8_k_compat | 5262 | 54293 | 1592192256 | 65536 | 4.5864759 | 1964748800 |

## Resultado

- Persistent reduce exactamente `480` malloc medianos y `672` frees medianos en cada celda medida.
- Persistent reduce `7,774,208` bytes solicitados medianos en cada celda medida.
- El scratch persistente retiene `65,536` bytes de capacidad pico.
- Buffered no muestra aumento mediano de RSS en este slice; mmap sube sólo decenas de KiB respecto a su coste base alto.
- Outputs exactos se conservan dentro de cada combinación activación/backend: prompt IDs, selected token, final token, checksums finales, logits, K/V cache y `cache_readback_ok`.
- Wall-clock no mejora de forma material: F32 persistent es levemente más lento en este slice; `q8_k_compat` buffered queda neutro y mmap mejora levemente. No se afirma mejora global.

## Allocations restantes

Quedan allocations relevantes fuera del seam mínimo:

- spans QXF buffered para pesos, filas de expertos y chunks de `lm_head`;
- buffers de ejecución con lifetime real de run (`kbuf`, `vbuf`, caches K/V, escalas, residuals);
- allocations propias de tokenizer/JSON/proceso fuera del hot path scratch.

No se promueve `persistent` a default. El siguiente trabajo debe decidir con más prompts/tokens si conviene extender el workspace a spans buffered o dejar el beneficio como opt-in diagnóstico.

## Por qué no empezar prioridad 3

La prioridad 3 (`dequant+dot fusionado`) sigue bloqueada por disciplina de roadmap: Issue #25 sólo prueba reducción de allocation/free del seam scratch. No introduce evidencia nueva sobre kernels fused, SIMD, threading ni CUDA.
