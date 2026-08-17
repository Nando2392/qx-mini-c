# Security Policy

## Supported status

QX-mini-MoE is a research runtime and has no stable release branch yet.

## Reporting

Report security issues privately through GitHub Security Advisories for this repository. Do not open a public issue containing exploit details, credentials or private model data.

## Threat boundaries

- GGUF/QXF inputs are untrusted binary data. Parsers must validate sizes, ranks, offsets, overflows and file bounds.
- Model files may be very large; avoid attacker-controlled allocations and integer wraparound.
- Download scripts must use explicit trusted URLs and should support checksum verification.
- Never place API keys or access tokens in model metadata, tests, logs or the wiki.

Model weights are not part of this repository and retain their upstream licenses and security considerations.
