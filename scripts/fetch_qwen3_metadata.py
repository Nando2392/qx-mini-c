#!/usr/bin/env python
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "qwen3_targets.json"

CONFIG_REPOS = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
}

GGUF_REPOS = {
    "qwen3-4b": "Qwen/Qwen3-4B-GGUF",
    "qwen3-8b": "Qwen/Qwen3-8B-GGUF",
    "qwen3-14b": "unsloth/Qwen3-14B-GGUF",
    "qwen3-30b-a3b": "unsloth/Qwen3-30B-A3B-GGUF",
}

def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def main():
    result = {"source": "huggingface", "models": {}}
    for key, repo in CONFIG_REPOS.items():
        cfg = get_json(f"https://huggingface.co/{repo}/raw/main/config.json")
        slim = {k: cfg.get(k) for k in [
            "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
            "max_position_embeddings", "num_experts", "num_experts_per_tok",
            "moe_intermediate_size", "torch_dtype"
        ] if k in cfg}
        result["models"][key] = {"config_repo": repo, "config": slim, "gguf_repo": GGUF_REPOS.get(key), "gguf_files": []}

    for key, repo in GGUF_REPOS.items():
        tree = get_json(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true")
        files = []
        for item in tree:
            path = item.get("path", "")
            if item.get("type") == "file" and path.lower().endswith(".gguf") and not path.startswith("BF16/"):
                files.append({"path": path, "size_gib": round(item.get("size", 0) / 1024**3, 3)})
        files.sort(key=lambda x: x["size_gib"])
        result["models"][key]["gguf_files"] = files

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    for key, data in result["models"].items():
        print(key, data["config"])
        for f in data["gguf_files"][:6]:
            print("  ", f["path"], f["size_gib"], "GiB")

if __name__ == "__main__":
    main()
