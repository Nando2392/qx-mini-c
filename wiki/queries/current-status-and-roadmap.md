---
title: Current Status and Roadmap
created: 2026-08-17
updated: 2026-09-25
type: query
tags: [roadmap, runtime, qwen3-moe, risk]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Current status and roadmap

## Entregado

- Conversión GGUF tensor-copy a [[qxf-format]].
- Loader, checksums y decoders quant necesarios.
- Attention real completa de layer 0.
- [[moe-forward]] real completo de layer 0.
- State loop real 0→47 para un token: residual, attention normalizada, Q/K RMSNorm, RoPE/GQA, KV INT8 y MoE top-8 en las 48 capas.
- [[final-output-head]] completo: final RMSNorm, 151936 logits Q6_K, top-N y argmax.
- [[autoregressive-loop]] greedy multi-token: re-embedding, posición y KV persistente por layer.
- [[qwen3-tokenizer]] QXT2: paridad exacta para prompts fijos y prefill desde texto.
- Issue #22 COMPLETED: matriz separada de Unicode, render Qwen3 chat-template y secuencias greedy para prompts múltiples, publicada en `f8facc600e4df708af22b5ea0e230dc1cf783ad1` con fix CI `333ee3df1b71a07b2912475ab423bc4bad12836f`; GitHub Actions `32409146859` terminó SUCCESS. La paridad demostrada sigue limitada a los casos y modalidades registrados; no implica paridad global de logits/modelo.
- [[cpu-inference-baseline]] fail-closed: A/B F32/Q8_K con startup/model-load, prefill, decode, total, peak RSS, provenance SHA-256 y outputs deterministas por modo.
- [[accumulated-kv-snapshot-replay]] fail-closed: captura/restaura K/V y escalas por layer/posición, con token de continuación y manifiesto SHA-256.
- Hardening QXF fail-closed: manifest, ABI, directorio, dims, nombres, placement, overflow y filas exactas.
- Golden independientes para embedding, IQ4_XS y expertos IQ2_XS/IQ3_XXS/IQ2_S representativos.
- Golden independiente y dispatch CPU opt-in para experto down IQ3_S × Q8_K.
- Smoke check y suite pytest.

## Gate activo

```text
state loop real layers 0–47: GREEN
final norm + lm_head completo: GREEN
autoregresión multi-token correcta: GREEN
tokenizer parity para prompts fijos: GREEN
QXF corruption/legacy-clamp gate: GREEN
modo Q8_K CPU compatible: GREEN, opt-in
F32 continúa default: DECIDIDO
bisect MoE layer 0 por etapa/experto: GREEN
Q8_K IQ2_XS/IQ3_XXS CPU opt-in: GREEN
Q8_K IQ2_S/IQ4_XS expertos CPU opt-in: GREEN
Q5_K decode + Q5_K×Q8_K atención CPU opt-in: GREEN
atención layer 1 con mismo attn_norm/KV F16: GREEN
sensibilidad layer 1→2 clasificada: GREEN
Q6_K decode + Q6_K×Q8_K atención CPU opt-in: GREEN
sweep layer 2→47 y sensibilidad layer 46 clasificada: GREEN
final RMSNorm + lm_head Q6_K×Q8_K same-input: GREEN
layer 47 attention + MoE same-input: GREEN
layers 44–42 attention + MoE same-input: GREEN
layer 41 IQ3_S down + bloque same-input: GREEN post-fix
layers 40–0 attention + MoE same-input: GREEN (41/41)
replay híbrido residual F16 layers 0–47: GREEN
clasificación acumulación/amplificación/modalidad: GREEN
perturbación escalada layer 1: GREEN; respuesta no suave, top-8 estable, cruces de orden observados
snapshot/replay de KV acumulado: GREEN; baseline 3 posiciones == captura 2 + replay 1
baseline CPU A/B F32/Q8_K: GREEN local; prefill/decode/total/RSS separados
QXF mmap read-only opt-in: GREEN local WSL2; gate 2x2 exacto por modalidad, buffered permanece default
scratch persistente opt-in: GREEN local; gate 2x2x2 exacto por modalidad/backend, ephemeral permanece default
policy provenance gates #27–#42: GREEN local/CI
long-context measurement/report hardening #43–#65: COMPLETED y publicado
measured run aggregation hardening #66–#72: COMPLETED y publicado
full-logit reusable #74: COMPLETED y publicado
matriz CPU/read-only real #75: 3/3 greedy y 12/12 argmax; 0/12 tolerancia full-logit, paridad numérica refutada case-local
bisect de activación #76: amplificación material localizada en MoE layer 1; Q8_K cierra el seam same-input y mejora la matriz a 1/12 sin regresión greedy/argmax
bisect KV acumulado × activación #77: 6/6 diagonales exactas; 0/24 thresholds y 22/24 argmax; interacción token-dependiente localizada
replay residual con KV fijo #78: 2/2 controles exactos; Q8_K conserva `1318` con residual F32 en layer 2 y diverge de routing F32 desde layer 3
replay layer 3 con KV fijo #79: controles same-mode exactos; retiene `1318`, los IDs ordenados difieren de ambos baselines desde layer 3 y 0/2 comparaciones full-logit pasan thresholds
seams layer 3 #80: CLOSED en `b0c4019b493a2817b4c9d2219b917783266c77f9`; CI `35152297016` PASS; input común exacto, attention cambia `ffn_input`, routing integrado `89` vs `22`, fixed-input routing exacto
generación CPU nativa #81: CLOSED `0305290`; CI `35412907586` PASS; CLI texto→QXT→JSON y API C comparten loop F32/INT8-KV de 48 layers
→ #81 no implica paridad global, CUDA, cobertura/soak 4K, throughput ni release readiness
políticas CPU nativas #82: implementación local y medición real completas; 24/24 outputs exactos; release pendiente
→ `recommended_cells=[]`: ninguna celda satisface simultáneamente >=10% decode y RSS <=110%; defaults sin cambios
capacidad CPU nativa #83: CLOSED en `ec4d2fd`; CI `35674122057` PASS; medición real de 128 y 256 posiciones, una corrida por capacidad
→ API compatible conserva 64 posiciones; API caller-buffer admite <=4096, pero sólo 128/256 tienen ejecución real
gate CPU 4K #84: OPEN; una corrida excedió el deadline nativo de 8 h y salió 2 sin output ni reporte; gate NOT MET
→ RSS final y posiciones completadas desconocidos; 142.48 MiB a ~7 h 40 min es observación histórica no final
CUDA final-head #85: progreso opt-in verificado en una RTX 4070 Laptop (CC 8.9); CPU/F32 y policy `none` siguen default
→ driver fixed-v2: outputs `[1124,77]`, un upload, dos launches, cero fallbacks; copies raw byte-exactas con hashes raw/LF separados
→ actual-model current-source PASS: 151936 logits exactos con thresholds 0.001/0.0001/0.999999 y 6/6 outputs byte-identical
→ package local PASS: build/model/source hashes, logs completos, preflight, fault, runtime/memoria y regresiones verificados
→ RELEASE GATES NOT PASSED: faltan dos reviews independientes del staged digest exacto y los gates Auto Research/CI de la revisión final
```

El issue GitHub #7 quedó cerrado como validación completada en el commit `42b3fd8b76acc26efdc7c53b6e7b427825b56b95`. GitHub Actions `32064105028` pasó build, tests y wiki lint. El cierre significa que la hipótesis de paridad fue probada y refutada de forma reproducible; no significa que QX sea numéricamente idéntico a llama.cpp.

El gate [[llama-cpp-parity]] aisló layer 0, encontró y corrigió la falta de renormalización de pesos top-8. Tras el fix, layer-1 cosine sube a `0.999961` (F32/F16), pero exactitud residual/logit queda refutada. La primera diferencia aparece en `Vcur`: QX usa activación F32 y ggml usa activación temporal Q8_K para IQ4_XS.

El gate [[f32-vs-q8k-activation]] implementó `q8_k_compat` como modo CPU explícito. En `Vcur-0` reduce max-abs de `3.05e-4` a `7.45e-9`, usa 4672 bytes de workspace y, en ese baseline anterior, fue ~7.4% más rápido. Ese gate identificó `ffn_moe_out-0` como siguiente objetivo; el resultado supersedente está en [[moe-stage-bisect]].

El gate [[moe-stage-bisect]] usa el mismo `ffn_inp-0` en QX/llama.cpp y cierra router, top-8, pesos, gate/up, SwiGLU, down y mezcla de layer 0 dentro de max-abs `1.20e-6`. [[iq2-s-iq4-xs-q8k]] valida layer 1 con el mismo input: mezcla final max-abs `4.35e-5`, RMSE `9.62e-7`, cosine ≈`1`. [[layer1-layer2-sensitivity]] encontró un bug Q5_K real, añadió `Q5_K × Q8_K` y cerró atención layer 1 same-input. El sweep supersedente [[layer2-logits-sweep]] encuentra la siguiente amplificación en layer 46→47, añade `Q6_K × Q8_K` y reduce `ffn_inp-46` same-input de max-abs `0.00445557` a `1.19e-7`. End-to-end no mejora: layer-47 RMSE queda `0.0309398` y logits RMSE `0.0393805`; el bisect causal sitúa la amplificación en MoE (`3.663×`), con top-8 estable y experto 74 aportando `76.08%` del delta.

[[layer47-same-input]] cierra el último bloque con el `layer-47.f32` exacto: atención llega a `ffn_inp-47` con max-abs `6.10e-5`; la cadena attention→MoE reconstruye `l_out-47` con max-abs `2.57e-4`, RMSE `8.01e-6`, cosine ≈`1`, y top-8 exacto `[83,3,74,119,92,28,109,101]`. El delta dominante restante proviene de sensibilidad de reducción F32 del router multiplicada por outputs down grandes, no de un nuevo decoder quant roto. La divergencia global entra acumulada desde capas anteriores y no queda resuelta por este gate.

El bisect descendente [[layer41-iq3s-q8k]] cerró layers 44–42 y localizó en layer 41 el primer fallo material observado en esa ejecución: atención, routing y SwiGLU cerraban, pero `ffn_down_exps` IQ3_S seguía en `dequant_f32`. La matriz pytest versionada regenera sidecars y fija routing/métricas exactas para layers 24, 41, 42, 43 y 44. El golden real `IQ3_S × Q8_K` y el dispatch opt-in reducen down a max-abs `9.54e-7` y reconstruyen `l_out-41` con max-abs `4.76e-5`, RMSE `1.05e-6`, routing exacto `[48,73,69,18,96,104,88,26]`. El siguiente bisect empieza antes de layer 41; no se infiere todavía el origen acumulado global.

[[layers0-40-same-input]] completa ese intervalo: 41/41 bloques cierran con el residual exacto del oracle y routing exacto. Los máximos materiales son `Vcur=1.79e-7`, `kqv_out=1.91e-6`, weighted `1.83105e-4` y `l_out=2.32019e-4`/RMSE `5.12959e-6`; todos pasan. Treinta y siete capas conservan sólo warnings de router logits por encima de `2e-6`, sin cambio top-8 ni fallo downstream. La hipótesis de otro seam local material queda refutada para este input; la divergencia global requiere ahora un bisect de acumulación, no otro kernel especulativo.

[[hybrid-residual-replay-accumulation]] completa el bisect de acumulación con KV F16 modal-equivalente. El baseline desde layer 0 termina con RMSE `9.87305e-2`; inyectar sólo el residual exacto de layer 1 reduce el final a `7.69131e-6` (`12836.6×`), aunque el error entrante original en layer 1 era apenas `1.46772e-8`. Esto prueba error acumulado/global y amplificación posterior de una discrepancia microscópica de layer 0, sin abrir otro seam local material. El control F32 no cierra (`9.05740e-2` tras replay de layer 1) y queda clasificado como efecto modal. La inyección sustituye sólo residual: el KV lo recalcula QX y no equivale a replay de KV acumulado para secuencias multi-token.

[[scaled-layer1-residual-sensitivity]] completa la perturbación escalada sobre la dirección observada de layer 0. La respuesta no es suave: escalas exactas `-1` y `+1` tienen el mismo L2 de entrada (`6.64213e-7`) pero producen deltas finales de `5.73359e-4` y `4.46778`. Ninguna de las 15 escalas cambia la membresía top-8. Sólo aparecen cruces de orden dentro del mismo set en layers 46 y 28; están correlacionados con ramas de respuesta alta, pero no prueban causalidad. No se autoriza un fix numérico: el siguiente gate debe separar orden de acumulación, thresholds de activación y modalidad sobre más tokens.

[[scaled-residual-token-modality-matrix]] completa ese gate con 18 celdas: tokens `42,9707,0`, activaciones F32/Q8_K-compatible y KV F16/F32/INT8. Diecisiete celdas cambian orden y catorce cambian membresía en alguna escala, pero las tres celdas runtime-aligned Q8_K-compatible + F16 conservan la membresía top-8. La asimetría `-1/+1` es extrema para tokens `42` (`7792.3×`) y `0` (`105030×`) en ese slice, mientras `9707` queda en `1.54585×` sin transición. El resultado separa sensibilidad de token/modalidad; no autoriza un fix ni una conclusión multi-token.

[[accumulated-kv-snapshot-replay]] cierra el seam previo al experimento multi-token. El payload nativo `QXKVSNP1` v2 conserva K/V, escalas, geometría, posición, seed y siguiente token, y el importador C verifica su trailer SHA-256 antes de consumir el cache. El manifiesto externo fija modelo/binario/revisión, SHA-256 y cobertura exacta por `(layer,position,kind)`. El control sintético exige igualdad exacta entre baseline de tres posiciones y captura de dos + replay de una. Truncación, bytes extra, magic incorrecto, mutación K/V de igual longitud y token de continuación distinto fallan cerrados. El gate se publicó en `0fc21c697994782b1f393dabd198855ef5ab939f` con CI `32370138054` SUCCESS; habilita experimentos posteriores, pero todavía no prueba paridad ni causalidad multi-token.

[[accumulated-kv-multi-token-perturbation-matrix]] usa ese seam sobre un QXF real y separa por KV F16/INT8 una dirección sintética F32 controlada en la continuación tras dos posiciones acumuladas. Baseline/replay y escala cero cierran exactamente en ambos modos. Las seis corridas conservan token `56`; F16 `+1` cambia orden de routing y las otras tres perturbaciones firmadas no. El resultado prueba mecánica reproducible en el slice registrado, no causalidad semántica ni equivalencia modal. Issue #21 fue publicado/completado en `4066538e88ccc0a18fafa213b606e9e619a3f9b5`; GitHub Actions `32390221256` terminó SUCCESS.

`state-loop-probe --full-moe --final-head --steps 2` produce ahora `42 → 1124 → 50853`. El token `1124` se re-embebe en posición 1, cada una de las 48 capas atiende dos posiciones mediante KV INT8 persistente y ambos checksums de 151936 logits se validan con el helper Q6_K oficial.

La comparación externa secuencial fija queda GREEN post-Q5_K: QX F32/INT8 coincide con llama F16/Q8_0 en `[1124, 50853]` para `[42]`, y coincide con llama F16 en `[358, 1184]` para `Hello!`. Llama Q8_0 produce `[358, 614]`; cobertura exhaustiva y paridad exacta de logits siguen pendientes.

[[qxf-mmap-io]] añade un backend QXF read-only explícitamente opt-in sin alterar kernels. El JSON final de Issue #24 (`wiki/evidence/issue-24-qxf-mmap-baseline.json`) supersede números antiguos: el gate 2×2 preserva exactamente outputs buffered/mmap dentro de F32 y `q8_k_compat`; observa ratios de wall-clock total `2.12548×` y `1.96456×`, respectivamente, mientras mmap añade aproximadamente `2.35 GB` de peak RSS y prefill/decode nativos no demuestran mejora material. Buffered sigue siendo default y la evidencia no autoriza inferencias de throughput global.

[[persistent-scratch-buffers]] completa Issue #25 para la prioridad CPU 2. La política `--scratch-policy persistent` queda opt-in; `ephemeral` sigue default/control. En la matriz 2×2×2 con 1 warm-up y 3 mediciones por celda, persistent reduce `480` malloc y `672` frees medianos por activación/backend, retiene `65,536` bytes de scratch, conserva outputs exactos por modalidad/backend y no muestra mejora wall-clock material. No se inicia prioridad 3 desde este resultado.

Issue #34 añade `--sampling-policy none` como contrato fail-closed de provenance para separar greedy determinístico de futuros samplers. `none` queda default, `sampling_profile` reporta `mode=greedy`, `stochastic_samples=0`, `top_p_evaluations=0` y `beam_width=1`; políticas no soportadas fallan antes de prompt/model/tokenizer I/O. No implementa top-p/min-p/beam, no promueve default y no afirma speedup/calidad.

Issue #35 añade `--long-context-policy none` como contrato fail-closed para separar el baseline actual de futuros gates 4K/RSS/KV-quality/soak. `none` queda default, `long_context_profile` reporta contadores inactivos en cero; políticas no soportadas fallan antes de prompt/model/tokenizer I/O. No ejecuta benchmark 4K, no aplica límite RSS, no mide calidad KV/8h y no afirma speedup/calidad.

Issue #36 añade `--long-context-policy ctx4k-smoke` como primer gate de admisión 4K: exige `--ctx >= 4096` antes de prompt/model/tokenizer I/O, reporta `target_ctx_tokens=4096` y conserva RSS/KV-quality/soak inactivos. `none` sigue default. No mide throughput 4K, no promueve default y no afirma speedup/calidad.

Issue #37 añade `--long-context-rss-limit-bytes` como gate RSS opt-in para los experimentos `ctx4k-smoke`: default `0` queda deshabilitado, valores no cero sólo son válidos con `ctx4k-smoke`, y el harness falla cerrado si el `peak_rss_bytes` muestreado excede el límite. No instala límite duro de OS, no cambia allocator, no mide calidad KV/8h y no afirma speedup/calidad.

Issue #38 añade `--long-context-kv-quality-checks` como contrato fail-closed para futuros sweeps de calidad KV: default `0` queda deshabilitado y valores non-zero fallan antes de prompt/model/tokenizer I/O. No ejecuta sweep KV, no corre soak, no promueve defaults y no afirma calidad.

Issue #39 añade `--long-context-soak-seconds` como contrato fail-closed para futuros runners de soak long-context: default `0` queda deshabilitado y valores non-zero fallan antes de prompt/model/tokenizer I/O. No ejecuta soak, no promueve defaults y no afirma estabilidad.

Issue #40 conserva el `long_context_profile` validado dentro de cada compact-run del harness de benchmark. Es sólo provenance por medición: no ejecuta benchmark 4K nuevo, no cambia defaults, no implementa quality sweep/soak y no afirma rendimiento, calidad o estabilidad.

Issue #41 conserva el `long_context_profile` común dentro de cada summary del harness y falla cerrado si las mediciones de una celda mezclan perfiles. Es sólo consistencia de provenance: no ejecuta benchmark 4K nuevo, no cambia defaults, no implementa quality sweep/soak y no afirma rendimiento, calidad o estabilidad.

Issue #42 conserva el `long_context_profile` común a nivel top-level report y falla cerrado si las celdas de una matriz mezclan perfiles. Es sólo consistencia de provenance: no ejecuta benchmark 4K nuevo, no cambia defaults, no implementa quality sweep/soak y no afirma rendimiento, calidad o estabilidad.

Issue #43 añade `long_context_measurement` al reporte del harness para registrar ctx medido, número de celdas, número de runs y presencia de summary RSS. Para `ctx4k-smoke` exige `ctx >= target_ctx_tokens`. Es sólo metadata de medición reproducible: no ejecuta benchmark 4K pesado en CI, no cambia defaults, no implementa quality sweep/soak y no afirma rendimiento, calidad o estabilidad.

Issue #44 endurece `long_context_measurement` para rechazar `kv_quality_checks` y `soak_seconds` non-zero hasta que existan implementaciones reales. Es sólo hardening fail-closed de metadata: no ejecuta sweep KV, no corre soak, no cambia defaults y no afirma calidad o estabilidad.

Issue #45 añade `rss_limit_bytes` y `rss_limit_active` a `long_context_measurement` para conservar la provenance del límite RSS muestreado en reportes. Es sólo metadata report-level: no instala límite duro de OS, no cambia allocator y no afirma estabilidad.

Issue #46 endurece `long_context_measurement` para rechazar `rss_limit_bytes` negativo antes de marcar el límite como activo. Es sólo hardening report-level: no instala límite duro de OS, no cambia allocator y no afirma estabilidad.

Issue #47 endurece `long_context_measurement` para rechazar summaries RSS con `count <= 0` antes de reportar `measured_run_count`. Es sólo hardening report-level: no ejecuta benchmark nuevo, no cambia defaults y no afirma rendimiento o estabilidad.

Issue #48 endurece `long_context_measurement` para rechazar summaries RSS sin `count` antes de reportar `measured_run_count`. Es sólo hardening report-level: no ejecuta benchmark nuevo, no cambia defaults y no afirma rendimiento o estabilidad.

Issue #49 endurece `long_context_measurement` para rechazar `peak_rss_bytes` no-objeto antes de leer `count` o reportar `measured_run_count`. Es sólo hardening report-level: no ejecuta benchmark nuevo, no cambia defaults y no afirma rendimiento o estabilidad.

Issue #50 endurece `long_context_measurement` para rechazar reportes sin celdas antes de reportar `measured_cell_count` o `measured_run_count`. Es sólo hardening report-level: no ejecuta benchmark nuevo, no cambia defaults y no afirma rendimiento o estabilidad.

Issue #51 endurece la ruta de perfil/medición long-context para rechazar celdas benchmark no-objeto antes de leer `summary`. Es sólo hardening report-level: no ejecuta benchmark nuevo, no cambia defaults y no afirma rendimiento o estabilidad.

Issue #52 endurece la ruta de perfil/medición long-context para rechazar celdas benchmark que omiten `summary.long_context_profile` antes de derivar perfiles o metadata de medición. Es sólo hardening report-level: no ejecuta benchmark nuevo, no cambia defaults y no afirma rendimiento o estabilidad.

Issue #53 endurece la ruta de perfil/medición long-context para rechazar celdas benchmark que omiten `summary` antes de derivar perfiles o metadata de medición. Es sólo hardening report-level: no ejecuta benchmark nuevo, no cambia defaults y no afirma rendimiento o estabilidad.

Issue #54 endurece `long_context_measurement` para rechazar `rss_limit_bytes` non-zero cuando `policy=none` antes de reportar un gate RSS activo. Es sólo hardening report-level: no instala límite duro de OS, no cambia allocator/defaults y no afirma estabilidad.

Issue #55 endurece la agregación report-level para rechazar `summary.long_context_profile.enabled` distinto de `true` antes de derivar el perfil común o metadata de medición. No ejecuta benchmark nuevo, no cambia defaults y no afirma rendimiento o estabilidad.

Issue #56 endurece la agregación report-level para validar `disabled_reason` por policy en cada profile: `none_policy` para `none`, y null para `ctx4k-smoke`. La validación ocurre antes de comparar perfiles o derivar metadata; no ejecuta benchmark nuevo ni cambia defaults.

Issue #57 endurece la agregación report-level con una allowlist de policy por cada profile benchmark: sólo `none` y `ctx4k-smoke` son válidas. Policies ausentes/no soportadas fallan antes de profile drift o metadata de medición; no ejecuta benchmark nuevo ni cambia defaults.

Issue #58 endurece cada profile report-level para exigir enteros exactos non-negative en `target_ctx_tokens`, `rss_limit_bytes`, `kv_quality_checks` y `soak_seconds` antes de profile equality. Bloquea missing/non-int/bool/negative y no ejecuta benchmark nuevo ni cambia defaults.

Issue #59 valida el target exacto en cada profile report-level: policy `none` exige `target_ctx_tokens=0` y `ctx4k-smoke` exige `4096` antes de profile equality o measurement. No ejecuta benchmark nuevo ni cambia defaults.

Issue #60 valida RSS inactivo por cada profile report-level: policy `none` exige `rss_limit_bytes=0` antes de profile equality, mientras `ctx4k-smoke` conserva thresholds opt-in non-negative. No aplica límite duro de OS ni cambia defaults.

Issue #61 exige `kv_quality_checks=0` en cada profile report-level antes de profile equality o measurement. No ejecuta sweep de calidad KV, no cambia defaults y no autoriza claims.

Issue #62 exige `soak_seconds=0` en cada profile report-level antes de profile equality o measurement. No ejecuta soak runner, no cambia defaults y no autoriza claims de estabilidad.

Issue #63 exige que `ctx` sea un entero exacto positivo al construir `long_context_measurement`, antes de leer celdas o derivar metadata. No ejecuta benchmark nuevo ni cambia defaults.

Issue #64 exige que `cells` sea una lista exacta antes de emptiness, indexado, profile aggregation o measurement. Evita aceptar secuencias/mappings ambiguos y no cambia defaults.

Issue #65 aplica ese contrato de lista exacta directamente en `summarize_cells_long_context_profile`, cerrando invocaciones fuera del measurement gate sin cambiar defaults.

Issue #66 exige un run set no vacío en `summarize_runs` antes de leer el primer profile. Evita `IndexError`, no ejecuta benchmark nuevo y no cambia defaults.

Issue #67 exige que el contenedor de `summarize_runs` sea una lista exacta antes de emptiness o indexado. Evita shapes ambiguos y no cambia defaults.

Issue #68 exige objetos en cada run first/later antes de profile extraction/equality. Evita `AttributeError`, no ejecuta benchmark nuevo y no cambia defaults.

Issue #69 exige presencia de todos los campos métricos requeridos en cada run first/later antes de aggregation. Evita `KeyError`; type/value hardening queda pendiente y no cambian defaults.

Issue #70 exige tipo numérico nativo para cada métrica first/later antes de `float(...)`. Rechaza bool/string ambiguos; preserva los checks existentes de finitud/signo y no cambia defaults.

Issue #71 aplica el contrato report-level existente a cada profile first/later dentro de `summarize_runs` antes de equality. Rechaza metadata inválida con causa específica; no añade policy ni cambia defaults.

Issue #72 exige presencia explícita de `long_context_profile` en cada run first/later antes de shape/contract/equality. Distingue missing de null/non-object y no cambia defaults.

Issue #80 cerró y se publicó en `b0c4019b493a2817b4c9d2219b917783266c77f9`; GitHub Actions `35152297016` pasó. Con el mismo residual F32 de layer 2 y snapshot INT8 KV producido por F32, el input de layer 3 es byte-exact; la modalidad de attention cambia `ffn_input` (`max_abs 0.00958681`, RMSE `0.00130100`) y el routing integrado sólo difiere en el último ID ordenado (`89` vs `22`). Al fijar el `ffn_input` F32, ambos modos seleccionan exactamente los mismos IDs terminando en `89`, pero persisten diferencias en outputs de expertos y MoE (`max_abs 0.00210665`, RMSE `0.000659551`). Los controles `integrated_double` opt-in son byte-exactos; `legacy_f32` sigue default. El cierre no demuestra bug de kernel ni paridad global.

Issue #81 añade la ruta canónica de generación CPU nativa: `qxqxf generate` recibe QXF, QXT y prompt de texto, tokeniza, ejecuta el loop compartido de 48 layers con activación F32 y KV INT8, y devuelve JSON. La API C `qx_run_native_generation(...)` usa ese mismo loop. El gate real fija `Hello!` → prompt IDs `[9707,0]` → outputs reproducibles `[358,1184]`; controles de API paran en el primer token con EOS `358`, en el segundo con EOS `1184`, y no paran por EOS con `-1`. El binding rechaza vocabulario, fingerprint de payload, BOS, EOS o flags no canónicos antes de leer el modelo.

La API compatible `qx_run_native_generation(...)` conserva su contrato de resultado de 64 posiciones; la API caller-buffer `qx_run_native_generation_into_with_options(...)` admite presupuesto forward de hasta 4096 y `<= ctx`. Ese límite de admisión no es una corrida 4K ni cierra calidad KV/soak. El fixture de UTF-8 partido prueba sólo el decoder del tokenizer, no generación E2E. Issue #81 cerró en `0305290b3aba00d9db62f32caacfbc2e14cdbeb`; CI `35412907586` pasó. Evidencia y comandos: `wiki/evidence/issue-81-native-generation-report.json`.

Issue #82 mantiene sin cambios el JSON default y la API C compatible, y añade políticas opt-in de I/O, scratch, kernel y threading junto con `--execution-profile`. Fusión y pool se limitan al final head; no paralelizan attention ni MoE. El reporte real verificado (`SHA-256 ed801a0debc96d71dc7ea634cca434b37be0ed10024f84bb3bcfa2e53bda4125`) ejecuta 24 procesos: cuatro warmups y cinco mediciones por cada una de cuatro celdas. Los 24 producen exactamente IDs `[358,1184]`, texto `" I need"` y checksums full-logit `[13347842135191822952,6249376751730758761]`.

| Celda | Decode wall elapsed de fase MSVC, mediana ± MAD (s) | Wall E2E mediana ± MAD (s) | RSS muestreado mediana ± MAD (MiB) |
|---|---:|---:|---:|
| `baseline` | 21.280 ± 0.134 | 30.360 ± 0.173 | 19.422 ± 0.000 |
| `persistent_fused_serial1` | 22.565 ± 0.131 | 31.781 ± 0.084 | 19.422 ± 0.000 |
| `persistent_fused_pool2` | 21.251 ± 0.713 | 30.228 ± 1.030 | 261.867 ± 0.008 |
| `mmap_persistent_fused_pool2` | 17.326 ± 0.204 | 25.561 ± 0.056 | 2413.770 ± 0.008 |

La regla predeclarada exige conjuntamente ganancia decode >=10% más allá del MAD combinado y RSS <=110% del baseline. Ningún candidato pasa: `recommended_cells=[]` y no se promueve default. mmap+fused+pool reduce la mediana decode aproximadamente 18.58%, pero alcanza ~124.28× el RSS baseline. El RSS del proceso incluye páginas file-backed del mmap: no es heap ni RAM total del sistema. En MSVC, `clock()` mide wall elapsed dentro de los límites de fase, no CPU de proceso ni CPU acumulado de workers. Prefill excluye el último token del prompt; decode empieza procesándolo para producir el primer output y continúa con inputs generados. #82 está CLOSED en `3c8a634`; CI `35652844078` pasó. El reporte raw permanece inmutable; la corrección semántica autoritativa es [`issue-82-timing-semantics.json`](../evidence/issue-82-timing-semantics.json), con enlace a la fuente primaria de Microsoft.

Issue #83 añade `--capacity-profile` opt-in sin cambiar el JSON default, mantiene la API compatible de 64 posiciones y añade una API caller-buffer con admisión `<=4096`. El reporte real inmutable (`SHA-256 4fedd8cdf44496491e92a288304c5889b9f7a4634997d8892002158fe47ef91d`) sólo mide una corrida en 128 posiciones (127 prompt + 1 input generado) y una en 256 (255 + 1). Ambas producen IDs `[1124,264]` y texto `" \\ a"` desde prompts sintéticos de `a` repetida; no hay claim de calidad significativo.

| Posiciones reales | Prompt + input generado | Wall E2E | RSS pico muestreado |
|---:|---:|---:|---:|
| 128 | 127 + 1 | 18.159802 min | 24.699 MiB |
| 256 | 255 + 1 | 40.543901 min | 29.523 MiB |

Los counters demuestran posiciones consumidas, no influencia semántica del último token. Los inputs nativos congelados pre/post son iguales. `native_cpu_seconds` vuelve a estar mal rotulado: bajo MSVC `clock()` mide wall elapsed de fase; prefill excluye el último prompt token y decode lo incluye. El reporte raw no se modifica; [`issue-83-timing-semantics.json`](../evidence/issue-83-timing-semantics.json) corrige la semántica. Una sola corrida por capacidad no permite generalizar throughput. #83 está CLOSED en `ec4d2fd`; CI `35674122057` pasó.

Issue #84 intentó una corrida CPU real de 4096 posiciones sobre ese baseline: 4095 tokens de prompt + un input generado, F32/INT8-KV y políticas CPU default. El proceso excedió el deadline nativo de 28,800 s, salió `2` y dejó stdout/stderr vacíos, sin reporte ni conteo de posiciones completadas. Por tanto #84 sigue OPEN y el gate 4K está **NOT MET**. El timeout no demuestra imposibilidad de capacidad 4K ni identifica CPU compute como root cause.

El journal histórico fue sobrescrito por un bug post-experimento con `phase=prepare`, `status=failed`, `peak_rss_bytes=0` y `elapsed_seconds=28809.359`; no es una medición terminal nativa válida. RSS final queda desconocido (`null`). La observación padre de 142.48 MiB a ~7 h 40 min es sólo histórica/no final. [`issue-84-cpu4k-outcome.json`](../evidence/issue-84-cpu4k-outcome.json) conserva hashes y provenance de los spools raw inmutables. El runner ejecutado tenía SHA-256 `116212700099472958e0cbacc7cd433de0cdf93e7e16bf37dabb6b6e8fd25150`; el fix posterior/future-only que retiene journals terminales tiene SHA-256 `4bba43948f03d79b51b8e4a0503a84c12f796709de2f4d322bf3f8d6b2c12f6b` y 60 tests padre PASS en 3.81 s. No reescribe la preparación ni los journals originales.

Issue #85 implementa sólo el final head CUDA F32 opt-in: el build CPU conserva el stub no disponible y el runtime CUDA exige `--cuda-policy final-head-f32`. Desde la raíz del repositorio, `QX_CUDA_ROOT` (preferido) o `CUDA_PATH` debe apuntar al toolkit antes de ejecutar `build_cuda_msvc.bat cuda`; el fault runner admite además `python tests/run_cuda_final_head_faults.py --cuda-root <cuda-toolkit-root>`. La ruta privada real no se documenta. El preflight current-source `fresh-preflight-cudafuncattrs-20260925T184644-008440` pasó en una NVIDIA GeForce RTX 4070 Laptop GPU (CC 8.9), produjo `[1124,77]`, conservó output legacy profile-NULL `1124`, registró un upload, dos kernel launches y cero fallbacks. El stdout raw current-source y los hashes before/after están versionados en `wiki/evidence/issue-85-current-fixed-v2-*`; [`issue-85-artifact-manifest.json`](../evidence/issue-85-artifact-manifest.json) distingue SHA-256 de bytes ejecutados CRLF y SHA-256 normalizado a LF.

La corrida strict actual-model current-source más reciente es `build/issue85-acceptance/actual-model-20260925T192200-914a6125/acceptance-report.json`, SHA-256 `0cba74521a13cabf3c1003385e5865a413784c813f59e804c92f186b25f98d7c`. El manifest final-provenance SHA-256 `e572a629104dc6170e86474cd77db87ebf7a2a7de641b50fa88f775b3d1237f9` liga build fresco, logs completos, source CUDA `a7a43140358be5cc4950f5f0d670e303fb2ca422a383584e71e743eb2dfe5e59`, modelos y driver. Para un residual fijo de layer 47, los 151936 logits CPU/GPU son finitos y exactos: `max_abs=0`, RMSE `0`, cosine `1`, thresholds `0.001/0.0001/0.999999`, argmax `1124` y 6/6 outputs GPU byte-identical. Es evidencia same-input final-head, no paridad global.

El paquete local de publicación #85 queda **PASS**, pero los release gates siguen **NOT PASSED**: faltan dos reviews independientes sobre el staged digest exacto y después los gates Auto Research/CI de la revisión final. También pasan preflight en un dispositivo, 10 tests negativos fail-closed, runtime/memoria same-process acotado y suites solapadas (`780 passed, 3 skipped`; `76`; `36`; `61`, no se suman). La evidencia negativa inyecta retornos de error de APIs CUDA, incluido `cudaGetLastError` después de un launch real; no prueba un kernel realmente fallido ni recuperación de device loss. El desglose de performance/latencia es explícitamente nongating. Cobertura de más dispositivos, pruebas de kernel real/device loss y caracterización de performance quedan como limitaciones o trabajo futuro, no blockers de release de #85; no autorizan claims de velocidad ni recuperación de device loss. CPU 4K sigue siendo un gate separado de #84 y permanece NOT MET. No hay claim global, 4K, velocidad, release readiness ni promoción de defaults. Detalle: [`issue-85-cuda-final-head-report.json`](../evidence/issue-85-cuda-final-head-report.json).

## Después

1. Mantener #85 opt-in; congelar el staged digest exacto, obtener dos reviews independientes y después ejecutar los gates Auto Research/CI sobre la revisión final.
2. Tratar cobertura adicional de dispositivos, fallo real de kernel/device loss y desglose/medición de latencia como limitaciones nongating o trabajo futuro; la evidencia actual no autoriza claims de speedup, throughput sostenido, ausencia general de leaks ni recuperación de device loss.
3. Mantener CPU 4K como no probado: #84 no completó el gate. No repetir por inercia ni convertir el timeout en una conclusión causal.
4. Mantener diferidos resume nativo, quality sweep KV y soak/térmica hasta que cada uno tenga runner y evidencia propios.

## Riesgos

- Layout/padding de futuros GGUF.
- Tipos quant distintos por layer.
- Extrapolar probes parciales.
- Cache misses de expertos.
- Modelos locales grandes nunca deben entrar en Git.

Arquitectura: [[architecture]]. Evidencia: [[numerical-correctness]].
