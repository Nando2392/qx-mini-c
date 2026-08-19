# QX-mini-MoE Wiki

> Bóveda Obsidian y wiki compilada al estilo Karpathy. Leer primero esta página y [[current-status-and-roadmap]].
> Última actualización: 2026-08-19 | Páginas de conocimiento: 23

## Proyecto

- [[project-overview]] — objetivo, alcance y límites honestos.
- [[architecture]] — flujo GGUF → QXF → forward y memoria.
- [[qxf-format]] — contrato del contenedor QXF1.
- [[moe-forward]] — atención, MoE y carry real por las 48 capas.

## Correctitud

- [[final-output-head]] — final RMSNorm, head Q6_K completo y golden externo.
- [[autoregressive-loop]] — re-embedding greedy, posiciones y KV persistente multi-token.
- [[numerical-correctness]] — golden tests, referencias y gates.
- [[qwen3-tokenizer]] — QXT2, BPE Qwen2/GPT-2, goldens externos y prefill desde texto.
- [[auto-research-loop]] — ciclo hipótesis → experimento → evidencia → decisión.

## Comparaciones

- [[llama-cpp-parity]] — residuals/checkpoints/logits completos, bug de routing corregido y refutación de bit-parity.
- [[f32-vs-q8k-activation]] — ADR y evidencia del modo CPU Q8_K compatible frente al default F32.
- [[moe-stage-bisect]] — bisect Qwen3MoE por etapa/experto, kernels Q8_K y primera divergencia material.
- [[iq2-s-iq4-xs-q8k]] — gate independiente de expertos layer 1/47, tipos 22/23 y amplificación hacia layer 2.
- [[layer1-layer2-sensitivity]] — fix Q5_K, atención same-input y amplificación cuantificada por experto.
- [[layer2-logits-sweep]] — sweep 0–47, kernel Q6_K×Q8_K y sensibilidad MoE de layer 46.
- [[final-head-q6k-q8k]] — RMSNorm/head same-input, Q6_K×Q8_K completo y causalidad hasta logits.
- [[layer47-same-input]] — último bloque same-input, atención/MoE por etapa y sensibilidad F32 del router.
- [[layer41-iq3s-q8k]] — bisect 44→41, primer fallo down IQ3_S y cierre con Q8_K.
- [[layers0-40-same-input]] — replay exacto 40→0, 41/41 bloques cerrados y warnings de router separados.
- [[hybrid-residual-replay-accumulation]] — replay F16 de cada frontera, error acumulado frente a error local y amplificación causal.

## Rendimiento

- [[performance-model]] — bytes activos, techo físico y mediciones reales.
- [[optimization-priorities]] — orden de optimización CPU/CUDA.

## Estado

- [[current-status-and-roadmap]] — entregado, pendiente y próximo gate.

## Metadatos

- [[SCHEMA]] — reglas, taxonomía y política de evidencia.
- [[log]] — historial append-only de la wiki.
