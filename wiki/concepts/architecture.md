---
title: Architecture
created: 2026-08-17
updated: 2026-08-17
type: concept
tags: [runtime, qxf, memory, qwen3-moe]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Architecture

## Data plane

```text
GGUF
  └─ parser mínimo
      └─ tensor-copy QXF1
          ├─ header/manifest
          ├─ tensor directory
          └─ quant bytes preservados
```

## Decode plane previsto

```text
raw residual
→ attention RMSNorm
→ Q/K/V + per-head Q/K RMSNorm + RoPE + persistent KV
→ GQA causal attention
→ output projection + residual
→ FFN RMSNorm
→ router + top-8 experts
→ SwiGLU + down + weighted sum
→ next-layer residual
```

Layer 0 está ejercitado de extremo a extremo en un probe; el loop persistente multi-layer es el próximo gate de [[current-status-and-roadmap]].

## Memory hierarchy

- QXF completo en almacenamiento/page cache.
- Expert slices direccionables por offset.
- KV INT8 dinámico por vector.
- Caché de expertos prevista para RAM/VRAM.
- Backend actual usa lectura de archivo; [[optimization-priorities]] prioriza mmap.

Relacionados: [[qxf-format]], [[moe-forward]], [[performance-model]].
