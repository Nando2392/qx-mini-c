---
title: QXF Format
created: 2026-08-17
updated: 2026-08-17
type: concept
tags: [qxf, runtime, quantization]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# QXF format

QXF1 es el contenedor experimental de [[project-overview]]. Mantiene un manifest pequeño, un directorio de tensores y los bytes cuantizados del GGUF fuente.

## Manifest real

```text
magic=QXF1
model=qwen3_moe
layers=48
hidden=2048
q_heads=32
kv_heads=4
head_dim=128
experts=128
top_k=8
vocab=151936
tensors=579
```

## Quant types usados

- Q4_K: embeddings.
- IQ4_XS: Q/K/V y attention output.
- IQ2_XS / IQ2_S: gate/up de expertos según capa.
- IQ3_XXS / IQ3_S / IQ4_XS: down según capa.
- Q6_K: output/lm_head real del checkpoint.

## Invariantes

- Los offsets deben validar límites de archivo.
- El layout experto debe validar `dims[2] == expert_count`.
- No inferir slice size desde padding de un archivo futuro sin comprobarlo.
- Decoders nuevos requieren gates de [[numerical-correctness]].

Véanse [[architecture]] y [[moe-forward]].
