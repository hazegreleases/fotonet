# fotonet operations reference

A complete example application around fotonet rather than a single prediction snippet.
It covers safe checkpoint inspection, ordered class schemas, datasets, fresh training,
exact resume, image and streaming inference, AnchorPoint/BoxTransform geometry, track
lifecycle, event routing, durable JSONL output, validation, export, synchronized
benchmarking, health probes, configuration, CLI commands, and deterministic tests.

## Commands

```bash
fotonet-ops inspect weights/fotonetn.pt
fotonet-ops run --config config/application.toml --limit 300
python -m pytest
```

The repository intentionally does not bundle weights or publish AP claims. Use a trusted
native checkpoint and the canonical validation protocol for release evidence.
