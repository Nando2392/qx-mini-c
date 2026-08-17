# Contributing

Contributions are welcome while QX-mini-MoE remains experimental.

## Ground rules

- Correctness before speed.
- Add a failing behavioral test before production code.
- Never describe a partial probe as complete inference.
- Do not commit GGUF/QXF/model weights, secrets or generated binaries.
- Keep tensor types, dimensions, offsets and ownership explicit.
- Every allocation needs a matching release on all error paths.

## Development loop

```bash
cmd.exe /c build_msvc.bat
python -m pytest tests -q
python smoke_check.py
```

For numerical kernels, add an independent reference. Reusing runtime output as the expected value is not an independent gate.

## Performance changes

Provide:

- hardware and build flags,
- exact command,
- baseline and new result,
- correctness gate result,
- p50/p95 when latency-sensitive,
- clear statement of whether the number is a probe or full inference.

## Documentation

Update the Obsidian vault index and log when a durable architectural fact changes. Follow `wiki/SCHEMA.md` evidence labels.
