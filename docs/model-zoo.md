# Model Zoo

The alpha checkout includes the public nano checkpoint as `fotonet-n`.

## Alpha Checkpoint

| Checkpoint | Architecture | mAP@.50:.95 | Parameters | MACs at 640 | GFLOPs at 640 | Weights |
|---|---|---:|---:|---:|---:|---|
| `fotonet-n` | `fotonetn` | 22.68% | 2.75M | 2.43G | 4.85G | Included |

The MACs and GFLOPs numbers are for a 640x640 input. GFLOPs are estimated as two FLOPs per MAC.

Inference times are not out yet because the benchmark computer is IO bottlenecked, so the timing numbers are not reliable enough to publish.

## Public Architectures

| Name | Purpose | P2 | Quality Head | Status |
|---|---|---|---|---|
| `fotonetn` | Nano general detector | yes | yes | alpha checkpoint |
| `fotonetn0` | Edge nano detector without P2 | no | no | alpha architecture |
| `fotonets` | Small detector | yes | yes | alpha architecture |
| `fotonetm` | Medium detector | yes | yes | alpha architecture |
| `fotonetl` | Large detector | yes | yes | alpha architecture |
| `fotonetx` | Extra-large detector | yes | yes | alpha architecture |

## Development Weights

Training checkpoints and comparison weights are development artifacts. Keep them out of the public surface.
