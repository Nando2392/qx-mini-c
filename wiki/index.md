# QX-mini-MoE Wiki

> Bóveda Obsidian y wiki compilada al estilo Karpathy. Leer primero esta página y [[current-status-and-roadmap]].
> Última actualización: 2026-08-17 | Páginas de conocimiento: 14

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

## Rendimiento

- [[performance-model]] — bytes activos, techo físico y mediciones reales.
- [[optimization-priorities]] — orden de optimización CPU/CUDA.

## Estado

- [[current-status-and-roadmap]] — entregado, pendiente y próximo gate.

## Metadatos

- [[SCHEMA]] — reglas, taxonomía y política de evidencia.
- [[log]] — historial append-only de la wiki.
