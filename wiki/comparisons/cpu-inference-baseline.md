---
title: CPU Inference A/B Baseline
created: 2026-08-20
updated: 2026-08-20
type: comparison
tags: [performance, cpu, baseline, q8-k, f32]
sources: [state-loop-probe, issue-23]
confidence: high
---

# CPU inference A/B baseline

Issue: [#23](https://github.com/Nando2392/qx-mini-c/issues/23)

## Decisión

El baseline CPU compara el modo default F32 con `q8_k_compat` opt-in sin cambiar kernels numéricos. Mantiene fijos modelo, QXF, tokenizer, prompt, revisión y argumentos, separa startup/model-load, prefill, decode, latencia total y peak RSS, y conserva repeticiones crudas más mediana/MAD/desviación.

La evidencia acreditada es [`wiki/evidence/issue-23-cpu-baseline.json`](../evidence/issue-23-cpu-baseline.json), SHA-256 `67bc4b8613d5e959ef04885fc644b35055d228e5c538ba56534002513422e816`.

## Contrato fijo

- revisión fuente: `333ee3df1b71a07b2912475ab423bc4bad12836f`;
- harness `scripts/q8k_perf_experiment.py` SHA-256: `358fdd58362b249a62e6af6e2ed9da4ac48c8043259caeb8de1d0623d42fb012`;
- binario MSVC SHA-256: `684e307a8682406d51a4ab3b495294424b0df451746a8a22c58fb677d71cbc46`;
- GGUF fuente `Qwen3-30B-A3B-UD-IQ2_M`, 10,865,578,560 bytes, SHA-256 `c8c2dc330dd1ec0c72c31b12e318647e6f9e0c773b9123eccfc3d12d9acc6652`;
- QXF `Qwen3-30B-A3B-UD-IQ2_M`, 10,860,081,152 bytes, SHA-256 `5609589e45a610bee6699f336109f3231326850d8f1ca839c614667c2f439840`;
- tokenizer QXT SHA-256: `8ec26e5ae058b271c0e17ee28aea3cf4d2b9ae9a4cd8f68f7aac81e9380761c6`;
- prompt `Hello!`, tokens `[9707,0]`, prompt SHA-256 `334d016f755cd6dc58c53a86e183882f8ec14f52fb05345887c8a5edd42c87b7`;
- KV INT8, 48 layers, contexto 16, dos outputs, temperatura 0, seed 7;
- un warm-up y tres repeticiones medidas por modo.

`process_clock` mide prefill/decode dentro del runtime, incluida la emisión JSON de cada posición. El proceso completo se mide con reloj monotónico externo. Peak RSS se muestrea cada 5 ms. Startup/model-load es un `inspect` separado (apertura y validación de metadatos QXF, no lectura eager de todos los tensores) y no se resta de la inferencia.

## Resultados reales

| Métrica (mediana) | F32 | q8_k_compat | A/B observado |
|---|---:|---:|---:|
| startup/model-load común | 0.0078939 s | 0.0078939 s | separado |
| prefill | 9.120 s | 1.914 s | 4.76489× |
| prefill tokens/s | 0.109649 | 0.522466 | — |
| decode, 2 tokens | 19.425 s | 5.145 s | 3.77551× |
| decode tokens/s | 0.102960 | 0.388727 | — |
| total proceso | 28.3303 s | 7.10160 s | 3.98929× |
| peak RSS | 20,402,176 B | 20,402,176 B | 0 B |

Dispersión mediana: prefill MAD F32 `0.111 s`, Q8_K `0.043 s`; decode MAD F32 `0.104 s`, Q8_K `0.068 s`; total MAD F32 `0.2711529 s`, Q8_K `0.1145268 s`. El JSON conserva min/max/pstdev y todas las corridas.

## Gate de outputs

Cada modo es determinista en warm-up y repeticiones: tokens, residual final, normalización, logits y checksums KV permanecen estables dentro de su celda.

Los modos no son numéricamente equivalentes y el contrato lo registra explícitamente:

- F32 selecciona `[358,1184]`;
- `q8_k_compat` selecciona `[358,614]`;
- checksums numéricos también difieren.

Esta divergencia cross-mode es permitida porque `q8_k_compat` sigue siendo un modo numérico opt-in sin paridad greedy global. No se oculta ni se convierte en equivalencia. Drift dentro de un mismo modo, cambio de prompt/provenance/modalidad, campos incompletos, NaN/Inf o fases ausentes fallan cerrados.

## Límite de claim

El speedup es sólo el observado para esta máquina, QXF, prompt, KV y revisión. No demuestra mejora global, throughput conversacional sostenido, paridad de logits/modelo, equivalencia end-to-end ni comportamiento de otro hardware. CUDA queda fuera de alcance.

Roadmap: [[current-status-and-roadmap]]. Prioridades: [[optimization-priorities]].
