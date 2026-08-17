# QXF Tensor Copy Smoke Test

## Verify tensor-copy from synthetic GGUF to QXF

Run this validation script to ensure the tensor copy operation preserves data integrity.

```bash
# Build if needed
cmd.exe /c build_msvc.bat

# Create test fixtures
python scripts/make_synthetic_gguf.py --out tests/fixtures/qwen3-30b-a3b-mini.gguf

# Copy tensors to QXF
cmd.exe /c "build\qxqxf.exe create-from-gguf-copy --in tests\fixtures\qwen3-30b-a3b-mini.gguf --model qwen3-30b-a3b --quant q2 --out models\qwen3-30b-a3b-copy.qxf"

# Inspect output
cmd.exe /c "build\qxqxf.exe inspect --in models\qwen3-30b-a3b-copy.qxf"
```

## Expected output

```json
{
  "magic": "QXF1",
  "version": 1,
  "model_type": "qwen3_moe",
  "quant_type": "q2",
  "layers": 48,
  "hidden": 2048,
  "intermediate": 6144,
  "q_heads": 32,
  "kv_heads": 4,
  "head_dim": 128,
  "vocab": 151936,
  "tensor_count": 6,
  "dir_offset": 4096,
  "data_offset": 8192,
  "file_size": 32768
}
```

## Run Python verification

```bash
python scripts/verify_qxf_copy.py \
  tests/fixtures/qwen3-30b-a3b-mini.gguf \
  models/qwen3-30b-a3b-copy.qxf \
  32768
```

Expected: `VERIFIED: QXF copy integrity check passed`

## Failure modes

1. **Buffer overflow in copy** — Exit code != 0, missing data
2. **Misaligned tensor** — QXF inspect shows unexpected `dir_offset` or `data_offset`
3. **Missing checksum** — `file_size` field doesn't match actual file size

## Quick shell for Windows

```batch
cmd.exe /c "build\qxqxf.exe create-from-gguf-copy --in tests\fixtures\qwen3-30b-a3b-mini.gguf --model qwen3-30b-a3b --quant q2 --out models\qwen3-30b-a3b-copy.qxf && build\qxqxf.exe inspect --in models\qwen3-30b-a3b-copy.qxf"
```