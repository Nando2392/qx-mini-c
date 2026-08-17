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

- El manifest se rechaza antes de uso si enumera tipos desconocidos, dimensiones base cero, `kv_heads > q_heads` o parámetros MoE incoherentes.
- Header y directorio usan el ABI QXF1 de 272 y 208 bytes; `dir_offset`, `data_offset` y cada tensor están alineados a 4096 bytes.
- El directorio debe quedar después del header y terminar antes de `data_offset`, sin overflow aritmético.
- Cada tensor tiene nombre no vacío, único y terminado en NUL; rank 1–4, dimensiones activas no cero e inactivas cero.
- Los rangos de datos no se solapan y se validan con resta (`size <= file_size - offset`), no con una suma vulnerable a overflow; el directorio puede mantener orden lógico y el loader ordena spans auxiliares para comprobar overlap.
- `byte_size` se deriva de dims y traits GGML checked; dtype, quant, group size y flags deben ser coherentes con el encoding soportado.
- El tamaño físico debe coincidir exactamente con `header.file_size`; el writer materializa el padding final declarado.
- Metadata-only es un modo explícito: `file_size == data_offset` y todos los tensores tienen `offset=byte_size=0`.
- Las rutas de embedding/lm_head exigen `byte_size % rows == 0`; no rebobinan a la fila 0 ni truncan spans.
- El layout experto debe validar `dims[2] == expert_count`.
- No inferir slice size desde padding de un archivo futuro sin comprobarlo.
- Decoders nuevos requieren gates de [[numerical-correctness]].

El gate de corrupción sintética cubre layout, dimensiones, traits, nombres duplicados,
offsets, solapamientos, spans cero, trailing bytes, divisores de manifest y clamps legacy. El
QXF real de Qwen3-30B-A3B sigue abriendo y ejecutando el golden completo.

Véanse [[architecture]] y [[moe-forward]].
