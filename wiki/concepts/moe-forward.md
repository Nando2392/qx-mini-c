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
→ selección top-8
→ renormalización de los ocho pesos seleccionados a suma 1.0
→ gate/up projections
→ SiLU(gate) × up
→ down projection
→ suma ponderada
→ residual final de layer N
```

La comparación independiente con Qwen3MoE en llama.cpp demostró que los ocho pesos seleccionados deben renormalizarse por su suma. QX omitía ese paso; el defecto fue corregido en las dos rutas reales y está cubierto por tests que exigen ocho expertos únicos y suma top-8 igual a `1.0`. Véase [[llama-cpp-parity]].

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
- Validado externamente: el fix de routing redujo el error de entrada de layer 1 desde max-abs aproximado `1.08` hasta `0.00589` en F32/F16; la paridad exacta restante está refutada por diferencias de contrato numérico posteriores.
- Pendiente: golden end-to-end de todos los tipos quant multi-layer.
- El modo attention `q8_k_compat` deja `ffn_inp-0` casi idéntico al oracle (max-abs `4.89e-5`), pero `ffn_moe_out-0` vuelve a divergir (max-abs `0.00792`). El próximo bisect debe separar gate/up, SwiGLU y down por experto; no atribuir esta diferencia al KV. Véase [[f32-vs-q8k-activation]].

Gates: [[numerical-correctness]]. Rendimiento: [[performance-model]].
