---
title: Final Head Q6_K × Q8_K Same-Input Gate
created: 2026-08-18
updated: 2026-08-18
type: comparison
tags: [qwen3-moe, final-head, q6-k, q8-k, logits, oracle]
sources: [comparisons/layer2-logits-sweep.md, concepts/final-output-head.md]
confidence: high
---

# Final head Q6_K × Q8_K same-input gate

> Estado: GREEN como contrato causal acotado. Issue: #13. La paridad residual global exacta sigue refutada.

## Pregunta

Después de [[layer2-logits-sweep]], ¿la divergencia desde `l_out-47` hasta logits nace en RMSNorm final, en `output.weight`, o sólo se propaga desde el residual ya perturbado?

## Contrato fijo

- runtime QX propio; llama.cpp sólo es oracle read-only;
- oracle fijado: commit `768d2a481a99cb75ec9a03b95dadbd35e7acf496`;
- hidden `2048`, vocabulario `151936`;
- `output_norm.weight` F32 y `output.weight` Q6_K;
- F32 sigue siendo el default;
- `q8_k_compat` sigue CPU-only y opt-in;
- sidecars y modelos permanecen fuera de Git.

El nuevo `final-head-probe` acepta un residual F32 exacto, aplica RMSNorm final, ejecuta el vocabulario completo y escribe `final-norm.f32` y `logits.f32`. Rechaza tamaños distintos de 2048 floats, NaN/Inf, modos desconocidos, top-N fuera de rango, tensores incompatibles y fallos de escritura; si falla la segunda salida, elimina la primera para no dejar un conjunto parcial válido en apariencia.

## Causa

El head QX previo siempre ejecutaba:

```text
Q6_K → dequant F32 → dot con activación F32
```

La ruta CPU de ggml para `output.weight` ejecuta:

```text
Q6_K × Q8_K
```

Con el `result_norm.f32` exacto del oracle, las filas reales 0, 1124, 75968 y 151935 de `Q6_K × Q8_K` coinciden bit a bit con los logits llama. El camino dequantizado completo conservaba argmax 1124, pero quedaba en max-abs `0.160992` y RMSE `0.0331809`.

## Fix

La ruta opt-in ahora:

1. calcula RMSNorm final F32;
2. cuantiza los 2048 valores una sola vez a ocho bloques Q8_K;
3. reutiliza ese workspace para las 151936 filas Q6_K;
4. reporta `lm_head_kernel=q6_k_q8_k` y `activation_quantizations=1`.

La ruta default conserva `lm_head_kernel=dequant_f32` y `activation_quantizations=0`.

## Gate same-input

Entrada: `l_out-47.f32` exacto del oracle.

| Checkpoint | Count | Max abs | RMSE | Cosine |
|---|---:|---:|---:|---:|
| RMSNorm final | 2048 | 3.81470e-6 | 1.17205e-7 | 0.999999999999998 |
| logits Q6_K × Q8_K | 151936 | 2.38419e-6 | 4.91155e-7 | 0.999999999999927 |

Argmax QX/llama: `1124`. Este resultado cierra el head con input idéntico dentro de error F32.

## Resultado global

Con el residual producido por QX después de layer 47:

| Checkpoint | Max abs | RMSE | Cosine | Delta L2 |
|---|---:|---:|---:|---:|
| `l_out-47` | 4.68347 | 0.172092 | 0.999995599911 | 7.78800 |
| RMSNorm final | 0.102143 | 0.00917996 | 0.999979206311 | 0.415437 |
| logits | 0.132130 | 0.0257469 | 0.999967105223 | 10.0359 |

Frente al head dequantizado del issue #12, logits mejoran de RMSE `0.0393805` a `0.0257469` y max-abs `0.187882` a `0.132130`. Argmax permanece `1124`; la secuencia fija sigue `[1124, 50853]`.

RMSNorm atenúa el error residual. El head same-input cierra. Por tanto, la divergencia global restante entra al head desde `l_out-47`; no justifica reabrir RMSNorm final ni Q6_K.

## Comandos

```text
build\llama_reference_oracle.exe model.gguf oracle 42 47 f16 internals=47
build\qxqxf.exe final-head-probe --in model.qxf --residual oracle\l_out-47.f32 --out-dir qx --activation q8_k_compat --top-n 5
python scripts\compare_logits.py --qx qx\logits.f32 --llama oracle\logits.f32
```

## Veredicto

- **GREEN:** probe same-input y sidecars fail-closed.
- **GREEN:** RMSNorm final same-input.
- **GREEN:** vocabulario completo Q6_K × Q8_K same-input.
- **GREEN:** F32 permanece default y Q8_K permanece CPU-only opt-in.
- **MEJORA GLOBAL ACOTADA:** RMSE logits `0.0393805 → 0.0257469`.
- **NO demostrado:** paridad exacta de `l_out-47`, logits globales o prompts exhaustivos.

## Siguiente gate

Bisect same-input de layer 47 desde su input hasta `l_out-47`, separando atención y MoE sin reabrir los kernels Q5_K/Q6_K ni el head final ya cerrados.

Relacionado: [[layer2-logits-sweep]], [[final-output-head]], [[numerical-correctness]], [[current-status-and-roadmap]].
