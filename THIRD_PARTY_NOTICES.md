# Third-Party Notices

QX-mini-MoE is MIT-licensed. The repository also interoperates with or derives reference data from the following projects; their licenses remain in force.

## llama.cpp / GGML

- Project: https://github.com/ggml-org/llama.cpp
- License: MIT
- Use here: quantization layout research, generated IQ lookup-table reference data, and an optional external numerical test helper linked against a user-provided local llama.cpp build.

The optional helper is not built by the default project build or CI. No llama.cpp binary or library is redistributed in this repository.

## Qwen3 and Unsloth GGUF

- Model family: https://huggingface.co/Qwen
- GGUF source used during local development: https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF

Model weights are not included. Users must review and comply with the model card, upstream license and redistribution terms before downloading or distributing any model artifact.

## Test and CI dependencies

- pytest: MIT
- GitHub Actions maintained by their respective authors under their published licenses.
