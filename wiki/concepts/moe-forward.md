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

## Layer 0 implementado y carry a layer 1

```text
post-attention residual
→ RMSNorm con blk.0.ffn_norm.weight
→ router blk.0.ffn_gate_inp.weight
→ softmax de 128 logits
→ top-8 sin renormalización adicional
→ gate/up projections
→ SiLU(gate) × up
→ down projection
→ suma ponderada
→ residual final de layer 0
```

Qwen3 usa `norm_topk_prob=false`; por ello los ocho pesos seleccionados no tienen que sumar uno.

## Shapes layer 0

```text
router: 2048 → 128 F32
gate/up: [2048, 768, 128] IQ2_XS
down: [768, 2048, 128] IQ3_XXS
```

## Límites

- Implementado y ejercitado: layers 0 y 1 con residual real encadenado.
- Layer 0 usa gate/up IQ2_XS y down IQ3_XXS; layer 1 usa gate/up IQ2_S y down IQ4_XS.
- Medición del probe 0→1, un token: ~0.34 s en la máquina de desarrollo; no es tok/s de inferencia completa.
- Pendiente: propagar el residual por las 48 layers.
- Pendiente: golden end-to-end de todos los tipos quant multi-layer.

Gates: [[numerical-correctness]]. Rendimiento: [[performance-model]].
