---
title: MoE Forward
created: 2026-08-17
updated: 2026-08-17
type: concept
tags: [qwen3-moe, runtime, quantization, validation]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# MoE forward

## Forward real y carry por 48 capas

```text
post-attention residual
→ RMSNorm con blk.N.ffn_norm.weight
→ router blk.N.ffn_gate_inp.weight
→ softmax de 128 logits
→ top-8 sin renormalización adicional
→ gate/up projections
→ SiLU(gate) × up
→ down projection
→ suma ponderada
→ residual final de layer N
```

Qwen3 usa `norm_topk_prob=false`; por ello los ocho pesos seleccionados no tienen que sumar uno.

## Shapes de referencia

```text
router: 2048 → 128 F32
gate/up: [2048, 768, 128] IQ2_XS
down: [768, 2048, 128] IQ3_XXS
```

## Límites

- Implementado y ejercitado: layers 0–47 con residual real encadenado para un token.
- Tipos observados `(gate, up, down)`: `(17,17,18)`, `(17,17,21)`, `(17,17,23)`, `(22,22,21)` y `(22,22,23)`.
- Medición del probe de 48 capas, un token: ~8.50 s en la máquina de desarrollo; no es tok/s de inferencia completa.
- Los 47 enlaces adyacentes cumplieron `residual_output_checksum[N] == residual_input_checksum[N+1]`.
- Pendiente: golden end-to-end de todos los tipos quant multi-layer.

Gates: [[numerical-correctness]]. Rendimiento: [[performance-model]].
