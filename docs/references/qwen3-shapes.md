# Qwen3-30B-A3B Tensor Shapes

## Standard naming convention

Qwen3 MoE models use the following tensor naming patterns:

| Tensor | Role | Shape (hidden=2048) |
|--------|------|---------------------|
| `token_embd.normal` | Token embedding matrix | [vocab, hidden] |
| `token_embd.emb_scaling` | Embedding scaling factor | [hidden] |
| `output_norm.weight` | Final RMSNorm scale | [hidden] |
| `rotary_emb.inv_freq` | Rotary embedding frequencies | [hidden/2] |

## Per-layer tensors (indexed `layers.N`)

### Attention sublayer

| Tensor | Role | Shape |
|--------|------|-------|
| `layers.N.attn_norm.weight` | QKV attention RMSNorm | [hidden] |
| `layers.N.attn.k_proj.weight` | Key projection | [hidden, hidden] |
| `layers.N.attn.v_proj.weight` | Value projection | [hidden, kv_heads * head_dim] |
| `layers.N.attn.o_proj.weight` | Output projection | [hidden, hidden] |
| `layers.N.attn gate` (MoE) | Expert gating | [num_experts] |

### MLP sublayer (MoE)

| Tensor | Role | Shape |
|--------|------|-------|
| `layers.N.mlp_norm.weight` | MLP RMSNorm | [hidden] |
| `layers.N.mlpdown gate` | MoE router gate | [num_experts] |
| `layers.N.mlp.up_proj.weight` | Expansion (h->4h) | [hidden, 4*hidden] |
| `layers.N.mlp.down_proj.weight` | Contraction (4h->h) | [4*hidden, hidden] |

## Quantized shapes

For Q2_K quantized tensors, storage format:

- F32: 4 bytes per element
- F16: 2 bytes per element
- Q2_K: 2 bytes per element after QK-Coder pack

## Memory estimation

For Qwen3-30B-A3B on 48 layers, hidden=2048:

```
Total parameters ≈ vocab * hidden + 2 * layers * (3 * hidden^2 + hidden * 4*hidden)
                 ≈ 313M weights + 48 * (3 * 4M + 8M * 4)
                 ≈ 307M * ~6 tensors per layer ≈ 1.8B parameters actual
```

Q2_K quantization: ~2 bytes/element → ~3.6GB for weights alone.

## Mini fixture tensors

For synthetic testing, 6 minimal tensors suffice:

```python
tensors = [
    {'name': 'token_embd.normal', 'dims': [151936, 2048], 'type': 'F16'},
    {'name': 'output_norm.weight', 'dims': [2048], 'type': 'F32'},
    {'name': 'layers.0.attn_norm.weight', 'dims': [2048], 'type': 'F32'},
    {'name': 'layers.0.attn.k_proj.weight', 'dims': [2048, 2048], 'type': 'F16'},
    {'name': 'layers.0.attn.v_proj.weight', 'dims': [2048, 512], 'type': 'F16'},
    {'name': 'layers.0.attn.o_proj.weight', 'dims': [2048, 2048], 'type': 'F16'},
]
```