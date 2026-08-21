---
title: QXF Buffered vs Memory-Mapped I/O
created: 2026-08-21
updated: 2026-08-21
type: comparison
tags: [performance, cpu, qxf, mmap, io, windows]
sources: [issue-24, qx_format.c, q8k_perf_experiment.py]
confidence: high
---

# QXF buffered vs memory-mapped I/O

Issue: [#24](https://github.com/Nando2392/qx-mini-c/issues/24)

## Hipótesis causal

El loader QXF conserva un `FILE *` abierto y valida header, tamaño físico, directorio, offsets, dimensiones, spans y solapamientos durante `qx_open_file`. Antes de este cambio, los pesos se consumían mediante `seek + fread`; las proyecciones recorrían ventanas de hasta 16 filas, copiándolas a un buffer temporal antes del dequant/dot.

Un mapping read-only puede eliminar esos `seek/read` repetidos y la copia de las ventanas de proyección. No cambia el coste dominante de dequantización, dot products, attention, MoE ni final head. Por tanto, la hipótesis es limitada:

- puede afectar apertura/acceso a pesos y latencia wall-clock total;
- puede afectar prefill/decode sólo por la porción de I/O/copia;
- puede cambiar peak RSS por la contabilidad de páginas file-backed;
- no implica acelerar kernels compute-bound ni mejorar resultados numéricos.

## Call path y diseño

```text
CLI --io-backend buffered|mmap
  -> qx_set_io_backend
  -> qx_open_file
       -> validación buffered de header/directorio/tamaño físico
       -> mmap read-only opt-in del archivo validado
  -> qx_acquire_span
       -> buffered: ownership heap + seek/read
       -> mmap: vista read-only sin ownership heap
  -> qx_projection_matvec_fill_mode
       -> buffered: ventana reusable + read/copy
       -> mmap: span directo validado
  -> qx_release_span
  -> qx_close_file
       -> unmap view -> close mapping handle -> fclose -> free directory
```

Buffered continúa siendo el default y control. Un `mmap` solicitado que no puede establecerse falla cerrado; no se etiqueta silenciosamente una corrida buffered como mmap. En Windows se usan `CreateFileMappingW(PAGE_READONLY)` y `MapViewOfFile(FILE_MAP_READ)`; POSIX usa `mmap(PROT_READ, MAP_PRIVATE)` para permitir un gate ejecutable alternativo bajo WSL sin relajar Smart App Control.

## Contrato de seguridad

Antes de exponer un span se rechazan tamaño cero, `offset > file_size` y `size > file_size - offset`; esta última forma evita overflow de `offset + size`. La apertura ya rechaza tamaño físico distinto al declarado, truncación, spans fuera del archivo y layouts corruptos. Todos los handles/views se inicializan a cero y `qx_close_file` centraliza cleanup en éxito y error. Un span deja de ser válido al liberarlo o cerrar su `qx_file` propietario.

No se mapea la directory: sigue copiada y validada en heap. Esto minimiza el cambio de ownership y conserva el formato QXF. No se modifican kernels, activaciones F32/Q8_K, threading, SIMD ni CUDA.

## Experimento 2x2

El harness compara en el mismo binario y entorno:

1. buffered + F32;
2. mmap + F32;
3. buffered + `q8_k_compat`;
4. mmap + `q8_k_compat`.

Mantiene el modelo/prompt del [[cpu-inference-baseline]], un warm-up y tres repeticiones medidas por celda. Registra startup/model-load separado por backend, prefill, decode, total, tokens/s, peak RSS, valores crudos, mediana, MAD, desviación y hashes SHA-256 de GGUF, QXF, tokenizer, prompt, ejecutable y script. El gate exige determinismo dentro de cada celda e igualdad exacta buffered/mmap dentro de cada modalidad. La divergencia F32/Q8_K sigue permitida y documentada.

Evidencia: `wiki/evidence/issue-24-qxf-mmap-baseline.json` (SHA-256 `52377c92d1b12d2b8265ac0b93b5ee779994a295af44de834aacc138dd9d1695`). El run WSL2 x86-64 fijado pasó equivalencia exacta buffered/mmap en F32 y `q8_k_compat`.

| Activación | Total buffered | Total mmap | Ratio buffered/mmap | Prefill ratio | Decode ratio | Delta RSS mmap-buffered |
|---|---:|---:|---:|---:|---:|---:|
| F32 | `64.4627 s` | `30.3285 s` | `2.12548×` | `0.88401×` | `0.99778×` | `+2,344,894,464 B` |
| `q8_k_compat` | `33.3610 s` | `16.9814 s` | `1.96456×` | `0.75160×` | `0.95764×` | `+2,346,631,168 B` |

Startup/model-load aislado fue `0.007432 s` buffered y `0.006519 s` mmap en este probe caliente. Mmap redujo el wall-clock total, pero elevó RSS en aproximadamente `2.35 GB`; prefill y decode nativos fueron más lentos o prácticamente iguales. La causa compatible con la evidencia es eliminar lecturas/copias repetidas del wall-clock fuera de los kernels a cambio de hacer residentes páginas del QXF; no es evidencia de mejora de kernels ni de throughput global. F32 y Q8_K conservaron sus secuencias distintas ya documentadas: `[358,1184]` y `[358,614]`.

## Limitaciones ambientales

Smart App Control bloquea ejecutables MSVC recién compilados con `WinError 4551`. El evento Code Integrity 3077 registra que `build/qxqxf.exe` no cumple el signing level de la política `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`. No se desactiva ni modifica esa protección y el resultado local Windows no se convierte en PASS.

La alternativa segura es compilar la misma fuente C17 con GCC dentro de WSL2 y ejecutar allí los contratos mmap y el benchmark 2x2. La compatibilidad Windows/MSVC se conserva mediante build `/O2 /W4` y CI Windows; la evidencia debe etiquetar su plataforma real.

## Límite de claim

La evidencia prueba únicamente el binario, modelo, prompt, plataforma y argumentos fijados. No prueba una mejora global, paridad de logits/modelo, equivalencia F32/Q8_K, rendimiento de otros discos/SO ni beneficio sobre kernels compute-bound.

Roadmap: [[current-status-and-roadmap]]. Prioridades: [[optimization-priorities]].
