---
title: Qwen3 Tokenizer and QXT2 Sidecar
created: 2026-08-17
updated: 2026-08-17
type: concept
tags: [runtime, qwen3-moe, validation, tokenizer]
sources: [raw/project/project-state-2026-08-17.md]
confidence: high
---

# Qwen3 tokenizer and QXT2 sidecar

## Contrato implementado

El tokenizer exacto para los prompts fijos usa la metadata real del GGUF:

```text
tokenizer.ggml.model = gpt2
tokenizer.ggml.pre = qwen2
tokenizer.ggml.tokens + token_type
tokenizer.ggml.merges
BOS/EOS + add_bos/add_eos
```

`scripts/export_qwen3_tokenizer.py` lee GGUF v3 con Python stdlib y escribe un sidecar binario `QXT2`. El runtime C carga ese sidecar, verifica tamaño y checksum FNV-1a64, aplica pre-tokenización Qwen2, codificación byte-level GPT-2 y BPE por menor merge rank. Python sólo participa en conversión; encode, decode y [[autoregressive-loop]] usan C.

`QXT2` mantiene tokenizer y pesos separados de [[qxf-format]]: cabecera fija, strings `model/pre`, entradas de vocabulario con tipo, y pares de merge con longitudes explícitas. No usa TSV ambiguo para el encoder.

## Goldens externos

Los IDs se obtuvieron con `llama-tokenize` compilado desde el checkout local de llama.cpp contra el GGUF real `Qwen3-30B-A3B-UD-IQ2_M.gguf`, sin BOS:

```text
"Hello, Qwen3!"          -> [9707, 11, 1207, 16948, 18, 0]
"¡Hola, 世界! café 🚀"    -> [39832, 68012, 11, 220, 99489, 0, 51950, 11162, 248, 222]
"  alpha\tbeta\n\ngamma  " -> [220, 8287, 2233, 1915, 271, 32214, 256]
ChatML specials          -> [151644, 872, 198, 9707, 151645]
```

La suite codifica y decodifica cada caso con QX y exige igualdad exacta de IDs y bytes UTF-8. También cubre QXT2 truncado, checksum corrupto, tipo de token inválido, UTF-8 inválido, ID fuera de rango, prompt mayor de 4096 bytes y decode de control sin modo especial.

## Integración con prefill

`prompt-state-loop-probe` tokeniza un archivo de texto y alimenta esos IDs al forward real. Para `Hello!`, dos tokens de salida greedy y contexto cuatro:

```text
prompt IDs: [9707, 0]
forward inputs: [9707, 0, 117268]
phases: [prefill, generate, generate]
selected: [null, 117268, 69336]
layers_run: 144
kv_appends: 144
logits checksums: [null, 18359823378288781632, 7341130597423700663]
```

Los `P-1` prefills intermedios no recorren `lm_head`; el último token del prompt produce el primer token de salida. Después, cada token seleccionado se re-embebe en la posición siguiente con KV INT8 persistente.

## Fail-closed y límites

- Sólo acepta `model=gpt2` y `pre=qwen2`.
- Máximo 4096 bytes de prompt, 1 000 000 entradas y 256 MiB de payload.
- Longitudes, offsets, tipos, conteos, checksum y trailing bytes se validan antes de usar datos.
- El modo integrado exige vocabulario `151936`, full MoE, head completo, temperatura cero y contexto suficiente.
- El modo integrado exige el fingerprint QXT2 `6140965799433681264`, derivado del GGUF objetivo verificado. Es un guard de compatibilidad FNV-1a64 no criptográfico, no una prueba de autenticidad.
- El clasificador Unicode implementa las clases necesarias para los goldens ASCII, Latin, CJK, whitespace, puntuación, símbolos y emoji. Codepoints fuera de sus rangos aceptados fallan en vez de aproximarse.
- No aplica automáticamente un chat template; los tokens especiales ChatML se reconocen sólo con `--parse-special`.

## Límite de honestidad

GREEN: paridad exacta de tokenización para los cuatro prompts fijos, round-trip, errores adversariales y alimentación de IDs al loop real. Pendiente: cobertura exhaustiva de todo Unicode/Qwen3, expansión automática de chat template y paridad residual/secuencia end-to-end contra otro runtime.

Provenance del oracle: llama.cpp `768d2a481a99cb75ec9a03b95dadbd35e7acf496`; GGUF SHA-256 `c8c2dc330dd1ec0c72c31b12e318647e6f9e0c773b9123eccfc3d12d9acc6652`. Los vectores viven en `tests/fixtures/qwen3-tokenizer-llama-cpp-goldens.json` y se vuelven a contrastar ejecutando `llama-tokenize` cuando está disponible.

Evidencia numérica: [[numerical-correctness]]. Estado: [[current-status-and-roadmap]].
