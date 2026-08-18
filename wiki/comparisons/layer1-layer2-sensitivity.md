---
title: Layer 1 to Layer 2 Sensitivity Bisect
created: 2026-08-18
updated: 2026-08-18
type: comparison
tags: [qwen3-moe, attention, q5-k, q8-k, sensitivity, oracle]
sources: [comparisons/iq2-s-iq4-xs-q8k.md, comparisons/moe-stage-bisect.md]
confidence: high
---

# Layer 1 to layer 2 sensitivity bisect

## Decisión

El issue GitHub #11 encontró un **bug corregible real** antes del MoE de layer 1 y, después de corregirlo, aisló la diferencia restante como **sensibilidad cuantizada esperada** para el contrato opt-in `q8_k_compat`.

El bug tenía dos partes:

1. el decoder Q5_K de QX no seguía el layout de `dequantize_row_q5_K` de ggml;
2. las proyecciones Q5_K de atención no tenían kernel `Q5_K × Q8_K` y caían a F32 aunque la ruta CPU de ggml usa Q8_K temporal.

El fix alinea high bits/grupos Q5_K con el oracle fijado y añade un vec-dot escalar `Q5_K × Q8_K` con las mismas escalas, mínimos y `bsums` que ggml. F32 sigue siendo el modo predeterminado; `q8_k_compat` sigue siendo CPU-only y opt-in.

## Oracle y TDD

Oracle read-only:

```text
C:/Users/fjmn2/Dev/llama.cpp-k3
commit 768d2a481a99cb75ec9a03b95dadbd35e7acf496
```

El ciclo RED→GREEN añadió:

- captura `attn_norm-1` al oracle (`internals_captured=19`);
- golden Q5_K independiente con checksum de los 256 floats completos en cuatro bloques de V/output contra `dequantize_row_q5_K` público;
- probe de atención layer 1 con exactamente el mismo `layer-1`/`attn_norm-1`;
- modalidad explícita `--kv f16|f32` y metadata `kv_format`;
- comparador fail-closed de sensibilidad por checkpoint, routing y experto;
- metadata exacta para las familias `IQ4_XS × Q8_K` y `Q5_K × Q8_K`.

## Atención layer 1 con input idéntico

Token `42`, KV F16 en QX y llama.cpp. El contexto F16 se aplica después de `Vcur`, antes de `attn_output`, igual que el oracle.

| Checkpoint | count | max_abs | RMSE | cosine |
|---|---:|---:|---:|---:|
| `attn_norm-1` | 2048 | 1.49012e-8 | 5.13771e-10 | 1.000000000000000 |
| `Vcur-1` | 512 | 8.38190e-9 | 2.78220e-9 | 0.999999999999967 |
| `kqv_out-1` | 4096 | 1.90735e-6 | 8.69279e-8 | 0.999999999963896 |
| `attn_out-1` derivado | 2048 | 6.33299e-8 | 1.50897e-8 | 0.999999999999811 |
| `ffn_inp-1` | 2048 | 5.96046e-8 | 1.52099e-8 | 0.999999999999978 |

Metadata ejecutada:

```text
projection_kernel=q5_k_q8_k
v_projection_kernel=q5_k_q8_k
output_projection_kernel=q5_k_q8_k
kv_format=f16
```

Esto cierra la atención de layer 1 con input idéntico. La divergencia E2E restante no puede atribuirse al decoder, layout o vec-dot Q5_K ya validados.

## Impacto end-to-end

Token `42`, QX Q8_K/F32-KV frente a llama F16. La tabla anterior del issue #10 queda superseded por este resultado post-fix:

| Checkpoint | Antes del fix max/RMSE/cos | Después del fix max/RMSE/cos | Mejora RMSE |
|---|---|---|---:|
| layer-1 input | 0.000750065 / 0.000141082 / 0.999998410 | 0.000750065 / 0.000141082 / 0.999998410 | 1.00× |
| layer-2 input | 28.2042 / 0.626792 / 0.999695728 | 0.294106 / 0.00660423 / 0.999999906 | 94.91× |
| logits | — / 1.48097 / 0.874416 | 0.171505 / 0.0346769 / 0.999936486 | 42.71× |

El máximo de layer 2 mejora `95.90×`. El argmax permanece `1124`. La paridad numérica exacta global sigue sin demostrarse.

El gate secuencial post-fix sí cambia de estado para la matriz fija: QX F32/INT8 produce `[1124, 50853]` para `[42]` y `[358, 1184]` para `Hello!`, igual que llama F16; llama Q8_0 difiere sólo en el segundo token de `Hello!` (`614`). Esto prueba dos prompts y dos tokens, no cobertura exhaustiva.

## Propagación de la perturbación

Comparación nominal/perturbada con el mismo runtime Q8_K en el MoE de layer 1:

| Checkpoint | max_abs | RMSE | delta L2 | Ganancia |
|---|---:|---:|---:|---:|
| layer input | 0.000750065 | 0.000141082 | 0.00638465 | 1.00× |
| attention output derivado | 0.00586081 | 0.000358505 | 0.0162241 | 2.541× desde layer input |
| FFN input | 0.00661087 | 0.000389406 | 0.0176225 | 2.760× desde layer input |
| MoE output | 0.293681 | 0.00658377 | 0.297948 | 16.907× desde FFN input |
| layer output / layer-2 input | 0.294106 | 0.00660423 | 0.298874 | 46.811× desde layer input |

El primer checkpoint que cruza el umbral de ganancia `2×` es `attention_output_derived`. Esto describe propagación E2E de una perturbación previa; no contradice el gate same-input de atención.

## Routing y experto dominante

Top-8 nominal y perturbado, sin drops, adds ni swaps:

```text
[68, 114, 55, 90, 0, 9, 28, 73]
```

El comparador valida que cada raw weight coincida con la probabilidad del experto seleccionado, que `weights_sum` coincida con la suma raw y que `weights_norm = raw/sum` con suma normalizada 1. En la perturbación real, raw-weight max-abs es `0.000261843`, normalized-weight max-abs `0.000501402` y weight-sum delta `0.000620872`; todas las consistencias pasan.

El experto `68` aporta `0.295067` de delta L2 ponderado, `99.0332%` del delta L2 total del MoE. Es el amplificador dominante, no el primer origen.

En su índice SwiGLU `711`:

```text
gate:   10.4506083 → 10.4674816   delta +0.0168734
up:    -16.7907772 → -16.7988853  delta -0.00810814
SwiGLU: -175.468765 → -175.837021  delta -0.368256
```

La recomputación independiente de `SiLU(gate) × up` da `-175.468758` y `-175.837022`. La predicción de primer orden del delta es `-0.368127`; el residual no lineal es sólo `-1.28801e-4`. No hay evidencia de corrupción de layout o memoria.

Un barrido de la perturbación con factores `0`, `0.25`, `0.5`, `1` y `2` mantiene exactamente el mismo top-8. El comportamiento no es perfectamente lineal porque Q8_K cambia de bins, pero todos los outputs son finitos, reconstruibles y coherentes con SwiGLU. El máximo MoE permanece en el índice `940`, dominado por el experto `68`.

## Rendimiento post-fix

Un token, 48 capas, KV INT8, cinco repeticiones warm:

| Activación | mediana | MAD | min–max | peak RSS mediano |
|---|---:|---:|---:|---:|
| F32 | 8.64031 s/token | 0.19512 s | 8.18795–8.83542 s | 5,627,904 B |
| Q8_K compat | 2.41209 s/token | 0.01540 s | 2.32203–2.46311 s | 5,627,904 B |

Speedup observado: `3.58208×`; reducción de latencia: `72.0832%`; delta RSS mediano: `0.0%`. Es latencia del probe instrumentado, no throughput sostenido.

## Veredicto fail-closed

- **Bug real:** Q5_K decode/vec-dot de atención; corregido y cubierto por goldens independientes.
- **Atención con input idéntico:** cerrada.
- **MoE con input idéntico:** ya cerrado por [[iq2-s-iq4-xs-q8k]].
- **Routing perturbado:** estable, mismo top-8.
- **Amplificación residual:** sensibilidad cuantizada esperada de atención + SwiGLU del experto 68.
- **Paridad secuencial fija:** GREEN para `[42]` y `Hello!` a dos tokens contra llama F16.
- **Paridad numérica global/exhaustiva:** todavía no demostrada; no promover Q8_K a default.

## Reproducción

```bash
cmd.exe /c build_msvc.bat
cmd.exe /c tests/build_llama_reference_oracle.bat
cmd.exe /c tests/build_ggml_reference.bat
python -m pytest tests/test_q5_k_decoder_oracle.py tests/test_moe_stage_oracle.py -q
python scripts/q8k_e2e_experiment.py --qx-exe build/qxqxf.exe --model models/Qwen3-30B-A3B-UD-IQ2_M.qxf --oracle <oracle-e2e> --out <temp>
python scripts/compare_layer_sensitivity.py --oracle-dir <oracle-internals> --qx-dir <qx-e2e> --nominal-moe-dir <nominal> --perturbed-moe-dir <perturbed> --layer 1
```

Modelos, EXE/OBJ/DLL, sidecars `.f32`, JSON de experimentos y barridos permanecen fuera de Git.

Roadmap: [[current-status-and-roadmap]]. Gate anterior: [[iq2-s-iq4-xs-q8k]]. Contrato general: [[f32-vs-q8k-activation]].
