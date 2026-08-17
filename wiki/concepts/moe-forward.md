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

## Layer 0 implementado

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

- Implementado y ejercitado: un layer real.
- Pendiente: propagar el residual por 48 layers.
- Pendiente: golden end-to-end de todos los tipos quant multi-layer.

Gates: [[numerical-correctness]]. Rendimiento: [[performance-model]].
