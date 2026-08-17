#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
URL='https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-UD-IQ2_M.gguf'
OUT='models/Qwen3-30B-A3B-UD-IQ2_M.gguf'
EXPECTED_GIB='10.119'
echo "Downloading Qwen3-30B-A3B UD-IQ2_M (~${EXPECTED_GIB} GiB)"
echo "URL: $URL"
echo "OUT: $OUT"
curl -L --fail --retry 10 --retry-delay 5 --connect-timeout 30 --continue-at - --output "$OUT" "$URL"
echo "Download complete"
python - <<'PY'
from pathlib import Path
p=Path('models/Qwen3-30B-A3B-UD-IQ2_M.gguf')
print(f'file={p}')
print(f'bytes={p.stat().st_size}')
print(f'GiB={p.stat().st_size/1024**3:.3f}')
PY
