---
title: Optimization Priorities
created: 2026-08-17
updated: 2026-08-23
type: comparison
tags: [performance, cpu, cuda, memory, roadmap]
sources: [raw/project/project-state-2026-08-17.md]
confidence: medium
---

# Optimization priorities

| Prioridad | Cambio | Razón | Gate |
|---:|---|---|---|
| 1 | mmap QXF | elimina seek/read/copy por span; Issue #24 en gate 2x2 | outputs buffered/mmap iguales por modalidad |
| 2 | buffers persistentes | Issue #25 reduce allocation/free del scratch de atención/MoE sin cambiar outputs | opt-in; RSS/outputs/counters |
| 3 | dequant+dot fusionado | Issue #26 reduce temporales final-head Q6_K con kernel opt-in | baseline/fused exactos; no default promotion |
| 4 | thread pool por filas/expertos | Issue #28 añade primer pool real opt-in sólo para filas del final-head F32; MoE/expertos siguen fuera de alcance | deterministic output; fail-closed policy |
| 5 | AVX2/FMA | Issue #29 añade primer gate opt-in `--simd-policy avx2-fma` para el dot F32 del final-head Q6_K, detrás de `--kernel-policy fused`, F32, thread serial y runtime CPU gates; la salida conserva reducción double determinística para equivalencia exacta | scalar default; logits checksum equivalence; no default promotion ni speedup no medido |
| 6 | expert cache | Issue #30 añade superficie fail-closed `--expert-cache-policy none` y `expert_cache_profile` para separar baseline sin cache de futuros hits residentes | no-cache default; fake hits rechazados; sin speedup no medido |
| 7 | CUDA híbrido | Issue #31 añade superficie fail-closed `--cuda-policy none` y `cuda_profile` para separar baseline CPU-only de futuros backends híbridos | CPU-only default; fake kernels/device bytes rechazados; sin speedup no medido |
| 8 | prefill GEMM | Issue #32 añade superficie fail-closed `--prefill-gemm-policy none` y `prefill_gemm_profile` para separar baseline escalar de futuros kernels GEMM de prefill | scalar/current prefill default; fake GEMM calls rechazados; sin speedup no medido |
| 9 | speculative/KV2 | Issue #33 añade superficies fail-closed `--speculative-policy none` y `--kv2-policy none` con provenance explícita antes de draft decode/KV2 real | greedy/current KV default; fake draft/KV2 counters rechazados; sin speedup no medido |
| 10 | sampling policy | Issue #34 añade superficie fail-closed `--sampling-policy none` y `sampling_profile` para separar greedy determinístico de futuros top-p/min-p/beam samplers | greedy default; fake stochastic/top-p counters rechazados; sin sampler real ni speedup/calidad no medida |
| 11 | long-context gate | Issue #35 añade superficie fail-closed `--long-context-policy none` y `long_context_profile` para separar baseline actual de futuros gates 4K/RSS/KV-quality/soak | ctx actual default; fake 4K/RSS/calidad/soak counters rechazados; sin benchmark 4K ni claim |
| 12 | ctx4k admission | Issue #36 añade `--long-context-policy ctx4k-smoke` como gate opt-in que exige `--ctx >= 4096` y reporta target 4096 | default sigue none; no RSS/KV-quality/soak; sin speedup/calidad |
| 13 | long-context RSS limit | Issue #37 añade `--long-context-rss-limit-bytes` como límite opt-in para el sampler RSS del harness cuando `ctx4k-smoke` está activo | default 0/deshabilitado; fail-closed si peak RSS muestreado supera el límite; sin límite OS duro ni speedup/calidad |
| 14 | long-context KV quality contract | Issue #38 añade `--long-context-kv-quality-checks` como contrato explícito fail-closed | default 0; non-zero rechazado antes de I/O; sin sweep KV, soak ni claim de calidad |
| 15 | long-context soak contract | Issue #39 añade `--long-context-soak-seconds` como contrato explícito fail-closed | default 0; non-zero rechazado antes de I/O; sin runner soak ni claim de estabilidad |
| 16 | long-context profile in benchmark runs | Issue #40 conserva el `long_context_profile` validado dentro de cada compact-run del harness | provenance por medición; sin benchmark 4K nuevo ni claims |
| 17 | long-context profile in benchmark summaries | Issue #41 conserva `long_context_profile` en summaries sólo si todas las mediciones acuerdan | fail-closed ante perfiles mixtos; sin benchmark nuevo ni claims |
| 18 | long-context profile in benchmark reports | Issue #42 conserva `long_context_profile` a nivel report sólo si todas las celdas acuerdan | fail-closed ante celdas mixtas; sin benchmark nuevo ni claims |
| 19 | long-context measurement metadata | Issue #43 añade `long_context_measurement` al reporte para registrar ctx medido, celdas, runs y presencia de RSS summary | metadata de medición; sin nuevo benchmark 4K pesado ni claims |
| 20 | long-context inactive measurement counters | Issue #44 endurece `long_context_measurement` para rechazar contadores futuros activos sin backend real | `kv_quality_checks`/`soak_seconds` deben ser 0; sin sweep KV ni soak runner |
| 21 | long-context RSS metadata | Issue #45 conserva `rss_limit_bytes` y estado activo dentro de `long_context_measurement` | provenance del límite muestreado; sin límite OS duro ni cambio allocator |
| 22 | long-context RSS metadata hardening | Issue #46 rechaza `rss_limit_bytes` negativo dentro de `long_context_measurement` | fail-closed report-level; sin límite OS duro ni cambio allocator |
| 23 | long-context measured-run count hardening | Issue #47 rechaza RSS summary con `count <= 0` dentro de `long_context_measurement` | fail-closed report-level; sin benchmark nuevo ni claims |
| 24 | long-context RSS count presence hardening | Issue #48 rechaza RSS summary sin `count` dentro de `long_context_measurement` | fail-closed report-level; sin benchmark nuevo ni claims |
| 25 | long-context RSS summary shape hardening | Issue #49 rechaza `peak_rss_bytes` no-objeto dentro de `long_context_measurement` | fail-closed report-level; sin benchmark nuevo ni claims |
| 26 | long-context measurement empty-report hardening | Issue #50 rechaza reportes sin celdas dentro de `long_context_measurement` | fail-closed report-level; sin benchmark nuevo ni claims |
| 27 | long-context benchmark cell shape hardening | Issue #51 rechaza celdas benchmark no-objeto antes de leer profiles/measurement | fail-closed report-level; sin benchmark nuevo ni claims |
| 28 | long-context benchmark profile presence hardening | Issue #52 rechaza celdas benchmark sin `summary.long_context_profile` antes de derivar profiles/measurement | fail-closed report-level; sin benchmark nuevo ni claims |
| 29 | long-context benchmark summary presence hardening | Issue #53 rechaza celdas benchmark sin `summary` antes de derivar profiles/measurement | fail-closed report-level; sin benchmark nuevo ni claims |
| 30 | inactive long-context RSS limit hardening | Issue #54 rechaza `rss_limit_bytes` non-zero cuando `policy=none` | fail-closed report-level; sin límite OS ni claims |
| 31 | long-context benchmark enabled-state hardening | Issue #55 rechaza profiles benchmark con `enabled` distinto de `true` | fail-closed report-level; sin benchmark nuevo ni claims |
| 32 | long-context disabled-reason hardening | Issue #56 valida `disabled_reason` por policy en cada profile benchmark | fail-closed report-level; sin benchmark nuevo ni claims |
| 33 | long-context policy allowlist hardening | Issue #57 rechaza policy ausente/no soportada en cada profile benchmark | fail-closed report-level; sin benchmark nuevo ni claims |
| 34 | long-context numeric profile hardening | Issue #58 exige enteros exactos non-negative por campo y celda | fail-closed report-level; sin benchmark nuevo ni claims |
| 35 | long-context target profile hardening | Issue #59 valida target exacto por policy y celda | fail-closed report-level; sin benchmark nuevo ni claims |
| 36 | inactive RSS profile hardening | Issue #60 exige RSS limit cero para `none` por celda | fail-closed report-level; sin límite OS ni claims |
| 37 | KV-quality profile hardening | Issue #61 exige `kv_quality_checks=0` por celda | fail-closed report-level; sin quality sweep ni claims |
| 38 | soak profile hardening | Issue #62 exige `soak_seconds=0` por celda | fail-closed report-level; sin soak runner ni claims |
| 39 | measured ctx hardening | Issue #63 exige `ctx` entero exacto positivo | fail-closed report-level; sin benchmark nuevo ni claims |
| 40 | measurement cells container hardening | Issue #64 exige lista exacta | fail-closed report-level; sin benchmark nuevo ni claims |
| 41 | profile aggregator container hardening | Issue #65 exige lista exacta | fail-closed report-level; sin benchmark nuevo ni claims |
| 42 | measured run emptiness hardening | Issue #66 rechaza run sets vacíos | fail-closed report-level; sin benchmark nuevo ni claims |
| 43 | measured run container hardening | Issue #67 exige lista exacta | fail-closed report-level; sin benchmark nuevo ni claims |
| 44 | measured run shape hardening | Issue #68 exige objetos first/later | fail-closed report-level; sin benchmark nuevo ni claims |
| 45 | measured run field-presence hardening | Issue #69 exige métricas requeridas | fail-closed report-level; sin type/value hardening ni claims |
| 46 | measured run metric-type hardening | Issue #70 exige int/float nativos, no bool/string | fail-closed report-level; finitud/signo preservados |
| 47 | measured run profile-contract hardening | Issue #71 valida profiles first/later antes de equality | reutiliza contrato existente; sin policy nueva |
| 48 | measured run profile-presence hardening | Issue #72 exige campo profile first/later | distingue missing de null/non-object |
| 49 | reusable full-logit comparison | Issue #74 extrae la comparación sidecar del CLI | preserva schema/thresholds; aún sin matriz ni claim nuevo |
| 50 | real case-local full-logit matrix | Issue #75 mide 3 casos × 2 pasos × 2 modalidades | 12/12 argmax; 0/12 tolerancia numérica; paridad refutada localmente |

## Estado tras el hardening report-level

Issues #27–#72 están completados y publicados. Sus perfiles y validadores son contratos fail-closed, no implementaciones de expert cache, CUDA, GEMM, speculative decoding, KV2, sampling, benchmark 4K, quality sweep ni soak.

El siguiente milestone no continúa el hardening por inercia: reutiliza la matriz CPU/read-only de Issue #22 y el comparador de logits existente para ampliar cobertura case-local con provenance fija. Debe separar llama F16, llama Q8_0 y QX por modalidad, registrar secuencias greedy y métricas de logits completas, y conservar `logit_parity`, `semantic_equivalence` y `global_modality_equivalence` como no afirmadas salvo evidencia futura independiente. CUDA, 4K pesado y promoción de defaults permanecen fuera de ese slice.

Issue #74 inicia ese milestone extrayendo `compare_logit_files(...)` del CLI existente. La API devuelve las mismas métricas, argmax, thresholds y verdict para que un runner posterior pueda consumirlas sin subprocess/JSON; este slice no ejecuta oráculos ni añade casos.

Issue #75 completa el milestone CPU/read-only inicial con ejecución real sobre artefactos fijados: 3/3 secuencias greedy pasan y los 12 argmax coinciden, pero ninguna de las 12 comparaciones full-logit pasa `max_abs <= 0.1`, `RMSE <= 0.1` y cosine `>= 0.99`. Por tanto, el resultado sostiene corrección greedy sólo para esos casos y refuta paridad numérica bajo esas tolerancias; no promueve defaults ni equivalencia global.

## No priorizar todavía

- CUDA Graphs sin backend CUDA.
- EAGLE sin forward multi-token eficiente ni draft head compatible.
- KV 2-bit a contexto 4K: ahorro limitado frente al gap escalar actual.
- Persistent kernels/PTX antes de Nsight.

El kernel scalar `IQ4_XS × Q8_K` ya existe como modo `q8_k_compat` y no pasa a default. El baseline reproducible [[cpu-inference-baseline]] preserva el A/B F32/Q8_K y separa startup, prefill, decode, total y RSS: en el slice fijado observa `4.76489×` en prefill, `3.77551×` en decode y `3.98929×` total. F32 selecciona `[358,1184]` y Q8_K `[358,614]`; por tanto no existe equivalencia cross-mode ni paridad global. Antes de SIMD/threading, mantener este gate fail-closed y exigir causalidad separada para cualquier cambio de kernel. Véase [[f32-vs-q8k-activation]].

La prioridad 1 se ejecuta en [[qxf-mmap-io]] como backend read-only opt-in. Buffered sigue default/control; el gate exige igualdad exacta buffered/mmap dentro de F32 y dentro de `q8_k_compat`. El cambio no autoriza iniciar buffers persistentes, threading, SIMD ni CUDA en el mismo issue.

La prioridad 2 se ejecuta en [[persistent-scratch-buffers]] como política `--scratch-policy persistent` opt-in. El default sigue siendo `ephemeral`. En la matriz 2×2×2 de Issue #25 persistent reduce `480` malloc y `672` frees medianos por celda, retiene `65,536` bytes de scratch y conserva outputs exactos dentro de cada activación/backend. El wall-clock queda mixto/no material; no se promueve a default ni autoriza prioridad 3.

La prioridad 3 se ejecutó en Issue #26 como `--kernel-policy fused` opt-in limitado al final-head Q6_K F32. El default sigue `baseline`; la evidencia está en `wiki/evidence/issue-26-fused-dequant-dot-baseline.json` y no autoriza fusionar projection/MoE ni promover default.

La prioridad 4 empezó en Issue #27 con contrato `--thread-policy serial --threads 1`, `thread_profile` y rechazo fail-closed. Issue #28 añade `--thread-policy pool --threads N` como opt-in limitado al row loop independiente de `output.weight` en final-head Q6_K F32. Serial sigue default; `q8_k_compat`, `threads < 2`, `threads > 64`, políticas no soportadas, y rutas sin `--final-head` fallan cerradas. La evidencia mínima está en `wiki/evidence/issue-27-thread-policy-serial-baseline.json` y `wiki/evidence/issue-28-thread-pool-final-head-baseline.json`. No autoriza paralelizar MoE/expertos ni promover defaults.

La prioridad 6 empieza en Issue #30 como contrato de provenance: `--expert-cache-policy none` es el único valor soportado, queda default, y el payload nativo expone `expert_cache_profile` con hits/misses/bytes en cero. Este slice no implementa cache residente ni autoriza speedup; sólo bloquea claims falsos y prepara el A/B futuro.

La prioridad 7 empieza en Issue #31 como contrato de provenance CUDA: `--cuda-policy none` es el único valor soportado, queda default, y el payload nativo expone `cuda_profile` con backend `none`, bytes de dispositivo/transferencias y launches en cero. Este slice no implementa CUDA, residency ni scheduler híbrido; sólo bloquea claims falsos y prepara el gate futuro.

La prioridad 8 empieza en Issue #32 como contrato de provenance prefill GEMM: `--prefill-gemm-policy none` es el único valor soportado, queda default, y el payload nativo expone `prefill_gemm_profile` con backend `none`, llamadas GEMM, tokens batched, filas fusionadas y bytes temporales en cero. Este slice no implementa GEMM, BLAS/CUDA ni batching de prefill; sólo bloquea claims falsos y prepara el gate futuro de TTFT.

La prioridad 9 empieza en Issue #33 como contrato de provenance speculative/KV2: `--speculative-policy none` y `--kv2-policy none` son los únicos valores soportados, quedan default, y el payload nativo expone `speculative_profile` y `kv2_profile` con backend/formato `none` y contadores cero. Este slice no implementa draft decode, aceptación speculative, compresión KV2 ni PPL/lossless gate; sólo bloquea claims falsos y prepara el contrato futuro.

La prioridad 10 empieza en Issue #34 como contrato de provenance de sampling: `--sampling-policy none` es el único valor soportado, queda default, y el payload nativo expone `sampling_profile` con modo `greedy`, sin muestras estocásticas, sin evaluaciones top-p y `beam_width=1`. Este slice no implementa top-p/min-p/beam search ni calidad/speedup; sólo bloquea claims falsos y prepara el gate futuro de samplers.

La prioridad 11 empieza en Issue #35 como contrato de provenance long-context: `--long-context-policy none` es default y el payload nativo expone `long_context_profile` con `target_ctx_tokens=0`, `rss_limit_bytes=0`, `kv_quality_checks=0` y `soak_seconds=0`. Ese slice no ejecuta benchmark 4K, no aplica límite RSS, no mide calidad KV ni soak 8h; sólo bloquea claims falsos y prepara esos gates.

La prioridad 12 empieza en Issue #36 como primer gate de admisión 4K: `--long-context-policy ctx4k-smoke` es opt-in, exige `--ctx >= 4096` antes de prompt/model/tokenizer I/O y reporta `target_ctx_tokens=4096` manteniendo `rss_limit_bytes=0`, `kv_quality_checks=0` y `soak_seconds=0`. No mide throughput 4K, no aplica RSS, no valida calidad KV y no promueve default.

La prioridad 13 empieza en Issue #37 como gate RSS opt-in para long-context: `--long-context-rss-limit-bytes N` permanece en `0` por default y sólo se acepta con `ctx4k-smoke`; el harness falla cerrado si su `peak_rss_bytes` muestreado supera `N`. No instala límites duros de OS, no cambia allocator, no ejecuta KV-quality sweep ni soak, y no promueve defaults.

La prioridad 14 empieza en Issue #38 como contrato explícito de calidad KV: `--long-context-kv-quality-checks` queda default `0` y cualquier valor non-zero falla cerrado antes de prompt/model/tokenizer I/O hasta que exista un sweep real. No mide calidad KV, no corre soak, no promueve defaults y no autoriza claims de calidad.

La prioridad 15 empieza en Issue #39 como contrato explícito de soak: `--long-context-soak-seconds` queda default `0` y cualquier valor non-zero falla cerrado antes de prompt/model/tokenizer I/O hasta que exista un runner de soak real. No corre soak 8h, no promueve defaults y no autoriza claims de estabilidad.

La prioridad 16 empieza en Issue #40 como persistencia de provenance long-context en el harness: cada compact-run preserva el `long_context_profile` validado junto a la medición. No ejecuta benchmark 4K nuevo, no cambia defaults, no implementa quality sweep/soak y no autoriza claims de rendimiento, calidad o estabilidad.

La prioridad 17 empieza en Issue #41 como consistencia de summaries long-context: `summarize_runs` preserva el `long_context_profile` común y falla cerrado si las mediciones mezclan perfiles. No ejecuta benchmark 4K nuevo, no cambia defaults, no implementa quality sweep/soak y no autoriza claims de rendimiento, calidad o estabilidad.

La prioridad 18 empieza en Issue #42 como consistencia report-level long-context: el reporte conserva el `long_context_profile` común sólo si todas las celdas del benchmark acuerdan y falla cerrado si una matriz mezcla perfiles. No ejecuta benchmark 4K nuevo, no cambia defaults, no implementa quality sweep/soak y no autoriza claims de rendimiento, calidad o estabilidad.

La prioridad 19 empieza en Issue #43 como metadata de medición long-context: el reporte añade `long_context_measurement` con `measured_ctx_tokens`, `measured_cell_count`, `measured_run_count` y presencia de summary RSS. Para `ctx4k-smoke` exige `ctx >= target_ctx_tokens`; no ejecuta benchmark 4K pesado nuevo en CI, no cambia defaults, no implementa quality sweep/soak y no autoriza claims de rendimiento, calidad o estabilidad.

La prioridad 20 empieza en Issue #44 como hardening de metadata de medición: `long_context_measurement` rechaza `kv_quality_checks` o `soak_seconds` non-zero hasta que existan implementaciones reales de sweep KV y soak runner. No ejecuta sweep, no corre soak, no cambia defaults y no autoriza claims de calidad o estabilidad.

La prioridad 21 empieza en Issue #45 como provenance RSS report-level: `long_context_measurement` registra `rss_limit_bytes` y `rss_limit_active` desde el perfil común. Esto sólo conserva evidencia del límite muestreado del harness; no instala límite duro de OS, no cambia allocator y no autoriza claims de estabilidad.

La prioridad 22 empieza en Issue #46 como hardening de esa provenance RSS: `long_context_measurement` rechaza `rss_limit_bytes` negativo a nivel report antes de derivar `rss_limit_active`. Esto no instala límite duro de OS, no cambia allocator y no autoriza claims de estabilidad.

La prioridad 23 empieza en Issue #47 como hardening del conteo medido: `long_context_measurement` rechaza summaries RSS con `count <= 0` antes de sumar `measured_run_count`. Esto no ejecuta benchmark nuevo, no cambia defaults y no autoriza claims de rendimiento o estabilidad.

La prioridad 24 empieza en Issue #48 como hardening de presencia del conteo RSS: `long_context_measurement` rechaza summaries RSS sin `count` antes de leer o reportar `measured_run_count`. Esto no ejecuta benchmark nuevo, no cambia defaults y no autoriza claims de rendimiento o estabilidad.

La prioridad 25 empieza en Issue #49 como hardening de forma del summary RSS: `long_context_measurement` rechaza `peak_rss_bytes` no-objeto antes de leer `count` o reportar `measured_run_count`. Esto no ejecuta benchmark nuevo, no cambia defaults y no autoriza claims de rendimiento o estabilidad.

La prioridad 26 empieza en Issue #50 como hardening de reportes vacíos: `long_context_measurement` rechaza matrices sin celdas antes de derivar `measured_cell_count` o `measured_run_count`. Esto no ejecuta benchmark nuevo, no cambia defaults y no autoriza claims de rendimiento o estabilidad.

La prioridad 27 empieza en Issue #51 como hardening de forma de celda benchmark: la ruta de perfil/medición long-context rechaza entradas de celda no-objeto antes de leer `summary`. Esto no ejecuta benchmark nuevo, no cambia defaults y no autoriza claims de rendimiento o estabilidad.

La prioridad 28 empieza en Issue #52 como hardening de presencia del perfil por celda: la ruta de perfil/medición long-context rechaza celdas benchmark que omiten `summary.long_context_profile` antes de derivar perfiles o metadata de medición. Esto no ejecuta benchmark nuevo, no cambia defaults y no autoriza claims de rendimiento o estabilidad.

La prioridad 29 empieza en Issue #53 como hardening de presencia del summary por celda: la ruta de perfil/medición long-context rechaza celdas benchmark que omiten `summary` antes de derivar perfiles o metadata de medición. Esto no ejecuta benchmark nuevo, no cambia defaults y no autoriza claims de rendimiento o estabilidad.

La prioridad 30 empieza en Issue #54 como hardening del límite RSS inactivo: `long_context_measurement` rechaza `rss_limit_bytes` non-zero cuando `policy=none` antes de reportar `rss_limit_active`. Esto no instala un límite duro de OS, no cambia allocator ni defaults y no autoriza claims de estabilidad.

La prioridad 31 empieza en Issue #55 como hardening del estado enabled: la agregación report-level rechaza `summary.long_context_profile.enabled` distinto de `true` antes de derivar el perfil común o `long_context_measurement`. Esto no ejecuta benchmark nuevo, no cambia defaults y no autoriza claims de rendimiento o estabilidad.

La prioridad 32 empieza en Issue #56 como hardening del motivo de desactivación: cada profile report-level con policy `none` exige `disabled_reason=none_policy`, y `ctx4k-smoke` exige `disabled_reason=null`, antes de comparar perfiles o derivar mediciones. Esto no ejecuta benchmark nuevo, no cambia defaults y no autoriza claims.

La prioridad 33 empieza en Issue #57 como allowlist report-level de policy: cada profile benchmark debe declarar exactamente `none` o `ctx4k-smoke`; valores ausentes/no soportados fallan antes de comparar profiles o derivar mediciones. Esto no ejecuta benchmark nuevo, no cambia defaults y no autoriza claims.

La prioridad 34 empieza en Issue #58 como hardening numérico por celda: `target_ctx_tokens`, `rss_limit_bytes`, `kv_quality_checks` y `soak_seconds` deben ser enteros exactos non-negative antes de comparar profiles. Esto bloquea missing/non-int/bool/negative, incluido `False == 0`; no ejecuta benchmark ni cambia defaults.

La prioridad 35 empieza en Issue #59 como target policy-specific por celda: `none` exige `target_ctx_tokens=0` y `ctx4k-smoke` exige `4096` antes de profile equality o measurement. No ejecuta benchmark, no cambia defaults y no autoriza claims.

La prioridad 36 empieza en Issue #60 como RSS policy-specific por celda: `none` exige `rss_limit_bytes=0` antes de profile equality; `ctx4k-smoke` conserva thresholds opt-in non-negative. No instala límites OS ni cambia defaults.

La prioridad 37 empieza en Issue #61: cada profile report-level exige `kv_quality_checks=0` antes de profile equality o measurement hasta que exista un sweep real. No mide calidad ni cambia defaults.

La prioridad 38 empieza en Issue #62: cada profile report-level exige `soak_seconds=0` antes de profile equality o measurement hasta que exista runner real. No ejecuta soak ni cambia defaults.

La prioridad 39 empieza en Issue #63: el measurement report exige `ctx` entero exacto positivo antes de leer celdas o derivar metadata. Bloquea bool/non-int/cero/negativo sin ejecutar benchmark nuevo.

La prioridad 40 empieza en Issue #64: `long_context_measurement` exige que `cells` sea una lista exacta antes de emptiness, indexado o agregación. Rechaza null, tupla, mapping y string con error contractual explícito.

La prioridad 41 empieza en Issue #65: el agregador compartido `summarize_cells_long_context_profile` aplica el mismo límite de lista exacta cuando se invoca directamente, antes de emptiness o indexado.

La prioridad 42 empieza en Issue #66: `summarize_runs` rechaza explícitamente un run set vacío antes de indexar el primer profile, reemplazando `IndexError` por error contractual.

La prioridad 43 empieza en Issue #67: `summarize_runs` exige una lista exacta antes de emptiness o indexado, rechazando null, tupla, mapping y string con error explícito.

La prioridad 44 empieza en Issue #68: cada run first/later debe ser objeto antes de extraer o comparar `long_context_profile`, reemplazando `AttributeError` por error contractual.

La prioridad 45 empieza en Issue #69: cada run first/later debe incluir todos los campos métricos agregados antes de aggregation, reemplazando `KeyError` por error contractual. Tipos y valores quedan fuera del slice.

La prioridad 46 empieza en Issue #70: cada métrica requerida first/later debe ser `int` o `float` nativo antes de normalización, rechazando bool y strings numéricos. Finitud/signo siguen en los summary validators existentes.

La prioridad 47 empieza en Issue #71: `summarize_runs` valida cada `long_context_profile` first/later con el contrato report-level existente antes de equality, impidiendo que metadata inválida se degrade a drift genérico.

La prioridad 48 empieza en Issue #72: cada run first/later debe contener explícitamente `long_context_profile` antes de shape/contract/equality, distinguiendo campo ausente de null o valor no-objeto.

Base cuantitativa: [[performance-model]]. Disciplina: [[auto-research-loop]].
