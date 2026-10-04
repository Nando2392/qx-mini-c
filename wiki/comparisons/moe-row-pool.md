---
title: Persistent MoE Row Pool
created: 2026-10-04
updated: 2026-10-04
type: comparison
tags: [cpu, performance, validation, roadmap]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Workers persistentes por filas MoE

## Contrato implementado

Issue #87 amplía [[moe-forward]] mediante `--thread-policy moe-pool --threads N`, separado de `pool` (sólo final head). Sólo admite Windows, buffered, activación F32, sin CUDA y 2–64 workers. Conserva serial/CPU/F32/cache-none y tamaños de ABI. Expertos, routing y mezcla continúan seriales; únicamente las filas gate/up/down se distribuyen. No modifica las dependencias token-major de prefill.

Cada generación crea su propio pool. El coordinador adquiere bytes packed y libera el borrow después de drenar todos los callbacks. Workers no acceden al LRU ni al FILE; escriben filas distintas manteniendo el orden de reducción double. El callback interno devuelve cero en éxito, mientras la API de dispatch devuelve uno en éxito. Tests independientes detectaron la inversión inicial de estas convenciones; la corrección fue comprobada ejecutando el modelo real.

En perfiles nativos opt-in, `workers_used` incluye workers MoE y `final_head_parallel_jobs` permanece exclusivo del head. El probe añade `moe_pool_profile` sólo con `moe-pool`; `jobs` cuenta dispatches matvec, no filas. JSON serial permanece sin nuevos campos MoE. No se afirma thread-safety global de la biblioteca.

## Evidencia local verificada

- Gate Q4_K packed independiente: fórmula sin usar decoder QX, comparación directa serial/pool/oráculo, dimensiones 1/255/256/511, filas 1/3/5/7, workers 2/3, cache none/resident, sentinelas y repetición. `tests/test_moe_rows_numeric.py`: 2 PASS tras el fix.
- Preflight API/pool/harness: 44 PASS en la ejecución conjunta del padre. Valida aceptación/rechazo de políticas, no equivale a ejecutar modelos en las cinco APIs.
- Modelo real: serial/moe-pool sobre prompt fijo con dos posiciones, más tests de harness, 33 PASS conjuntos. Este preflight no sustituye la matriz final congelada.
- La simulación de rama no Windows mediante MSVC no es una ejecución Linux.

## Aceptación y límites

Runner: `scripts/moe_pool_acceptance.py`. Matriz F32 serial/moe-pool por cache none/resident-packed, un warmup y dos medidas por celda; 12 procesos con dos posiciones cada uno. Se exige igualdad directa de bytes de logits, tokens/checksums, workers/jobs reales, hits residentes y hashes pre/post de inputs. Cada proceso tiene deadline de 300 segundos y cap de 2 GiB para working set del proceso principal: muestras durante ejecución y pico vitalicio consultado mediante el handle Win32 retenido. Las métricas se conservan separadas; una infracción final rechaza la corrida después de su salida. No es límite duro de memoria del SO ni garantía del pico agregado de un árbol. El driver auditado crea threads; cualquier proceso hijo observado viola el contrato y se limpia.

La matriz local `build/issue87-release-matrix-20261004T005131710429Z/report.json` pasó las 12 corridas (6 cache none y 6 resident-packed), con logits byte-exactos y tokens `[1124, 77]`, hashes pre/post iguales y pico máximo de raíz de 624599040 bytes. Cinco carreras de salida se recuperaron únicamente tras acreditar salida cero e identidad de raíz y consultar su pico final. SHA-256 del reporte crudo: `4349b8de7d559b8849bbf3ea88491c05ff49f6a8e6ec4025cb2dca99cafae2ae`. Los artefactos raw son locales y no se confunden con evidencia descargable del repositorio. Regresión final: 1050 PASS y 5 skips explícitos; build, smoke y wiki lint PASS. El harness pasó 63 tests con build aislado. La carrera previa del test de supervisión se corrigió sincronizando la excepción con `tree.ready`, sin cambiar el runtime 4K ni ejecutar su campaña.

Estado de publicación pendiente de dos reviews del digest exacto, AutoResearch y CI. No hay claim de speedup, paridad global, rendimiento 4K, I/O físico o promoción de default. [[current-status-and-roadmap]] conserva #84 OPEN/NOT MET sin repetir su campaña de ocho horas.
