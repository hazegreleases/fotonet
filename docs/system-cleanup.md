# Production-system cleanup specification

This document defines the repository boundary for the first public fotonet
release. It is a description of the current system, not a compatibility plan
for discarded experiments.

## Production boundary

The supported detector is the graph in `fotonet/models/v1/`: one explicit
backbone, neck, and dual-assignment NMS-free head. The public registry contains
exactly ten model IDs: N, S, M, L, and X, each with an optional `-p2` variant.
Aliases and arbitrary YAML-defined graphs are rejected.

Training uses one uniform-sampling protocol assembled in
`fotonet/training/protocols/v1/`. Evaluation uses the same validation and COCO
metrics implementation whether invoked during training or from `Fotonet.val()`.

## Removed code

The public tree must not contain executable implementations of:

- earlier backbone, neck, head, or graph generations;
- automatic architecture probing or metadata-free checkpoint reconstruction;
- adaptive sampling controllers, Orbit, ART, distillation, pseudo-label
  ingestion, stage profiling, epoch cutting, or experimental FOLD graphs;
- an NMS inference branch or an unsafe pickle-loading switch;
- public model-name aliases or compatibility-only CLI task names.

Tests that existed only to preserve those implementations are removed. Current
tests instead verify that unsupported graph identities and metadata-less
artifacts fail closed.

## Active checkpoint exception

The in-progress Nano run predates the final public schema. It is accepted only
when all of its fixed identity fields match the known production Nano graph and
all removed controller states are disabled or `None`. The launcher retains the
old *field names* needed to prove that condition. Those serialized fields do
not expose or reactivate the removed systems.

Every newly saved checkpoint carries independent versions for the checkpoint
format and training protocol, plus a canonical model ID and architecture
fingerprint. Tensor-only loading is mandatory.

This exception ends after the active run is converted into a released,
self-identifying checkpoint. It must never grow into general legacy probing.

## Weight publication

Official weights are currently training. When ready, they will be attached to
GitHub Releases with SHA256 checksums and canonical COCO evaluation evidence.
Weights do not belong in Git history. Automatic download hooks for canonical
model names may be added after stable release assets exist; until then, users
pass an explicit checkpoint path.

## Planned scale rebalance

Nano remains the present reference architecture. S, M, L, and X will be
rebalanced later, after the first release work, toward these parameter bands:

| Scale | Target | Allowed range |
|---|---:|---:|
| S | 2.20M | 2.02M–2.38M |
| M | 5.00M | 4.50M–5.50M |
| L | 11.40M | 10.40M–12.40M |
| X | 33.80M | 28.80M–38.80M |

These are future design targets, not current measurements or accuracy claims.
Each change requires new static measurements, runtime benchmarks, training,
and canonical validation before replacing the published table.

## Completion checks

The cleanup is complete only when:

1. public imports cannot load removed graph or training modules;
2. all ten canonical graphs build and run odd-sized inference forwards;
3. the active checkpoint passes launcher dry-run and resume-state
   reconstruction without a training step;
4. native and exported artifacts reject absent or conflicting identity data;
5. focused tests, the non-training regression suite, package build, `twine
   check`, installed-wheel smoke checks, and strict COCO preflight pass;
6. public documentation contains no local paths, unpublished accuracy claims,
   or claims that weights will never be released.

Canonical COCO evaluation itself is intentionally deferred until the finished
weight is available. No AP value may be inferred from architecture inspection
or an interrupted checkpoint.
