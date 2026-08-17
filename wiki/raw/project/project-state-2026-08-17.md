---
source_url: local-project://qx-mini-c
source_paths:
  - src/qx_format.c
  - include/qx_format.h
  - tests/test_real_iq4xs_projection.py
  - tests/test_real_qkv_golden.py
  - tests/test_projection_rows.py
  - smoke_check.py
  - docs/BOOTSTRAP_STATUS.md
ingested: 2026-08-17
sha256: 9efe7ac7ee1df1bdb56d5dd99d13282d7e6e92e3dec62342b44c06e3337248e9
---

# Project state captured on 2026-08-17

## Model and format

- Target: Qwen3-30B-A3B, 48 layers, hidden 2048, 128 experts, top-8.
- Source GGUF local: 10,865,578,560 bytes; intentionally excluded from Git.
- Generated QXF local: 10,860,081,152 bytes; intentionally excluded from Git.
- QXF has 579 tensors and preserves source quant bytes.

## Real path exercised

```text
embedding Q4_K
→ attention RMSNorm
→ Q/K/V IQ4_XS
→ Qwen3 split-half RoPE
→ KV INT8
→ GQA 32Q/4KV
→ causal softmax
→ context 4096
→ attention output IQ4_XS 4096→2048
→ attention residual
→ FFN RMSNorm
→ router F32 2048→128
→ top-8
→ gate/up IQ2_XS
→ SiLU(gate) × up
→ down IQ3_XXS
→ weighted MoE sum
→ layer-0 residual
```

## Evidence

- External llama.cpp decoder agrees with representative IQ2_XS/IQ3_XXS expert rows.
- Layer-0 full forward probe median measured at ~0.2085 s on the current CPU path.
- 48-layer extrapolation is ~0.10 token/s before final norm/lm_head/tokenizer; this is an estimate, not completed inference.
- Active model traffic derived from QXF directory: ~1.339 GiB/token including lm_head, before KV.
- Current backend is CPU C. CUDA is not implemented.

## Open gate

The persistent state loop still needs real attention+MoE residual propagation across layers, then 48 layers, final norm, complete lm_head and tokenizer-equivalent greedy token gates.
