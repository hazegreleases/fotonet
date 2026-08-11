# Models and measured runtime

Named models currently construct architectures with random weights. The first
official `fotonetn` weight is training now and will be released with a checksum,
canonical validation report, and download hook after verification. No AP result
is published yet.

## Current graph metrics at 640x640

The deployment graph is `eval() -> strip_o2m_for_inference() -> fuse()`.
FLOPs use `2 x MACs`. Full parameters include training-only O2M projections;
deploy parameters exclude them and include Conv-BN fusion. FP32 state tensors
sum the training graph's tensor payload, including buffers; this is not a
serialized checkpoint size and excludes optimizer/EMA state.

| Model | Train params | Deploy params | MACs | FLOPs | Conv2d | Raw output | FP32 state tensors |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `fotonetn` | 1,042,936 | 1,005,932 | 1.313G | 2.626G | 64 | `[1,8400,84]` | 4.01 MiB |
| `fotonetn-p2` | 1,081,936 | 1,035,908 | 1.691G | 3.381G | 87 | `[1,34000,84]` | 4.17 MiB |
| `fotonets` | 1,515,694 | 1,479,255 | 2.138G | 4.275G | 73 | `[1,8400,84]` | 5.83 MiB |
| `fotonets-p2` | 1,567,536 | 1,525,972 | 2.664G | 5.328G | 90 | `[1,34000,84]` | 6.04 MiB |
| `fotonetm` | 2,973,070 | 2,933,803 | 3.696G | 7.392G | 80 | `[1,8400,84]` | 11.41 MiB |
| `fotonetm-p2` | 3,064,064 | 3,019,508 | 4.615G | 9.230G | 97 | `[1,34000,84]` | 11.77 MiB |
| `fotonetl` | 5,174,534 | 5,132,403 | 6.606G | 13.213G | 84 | `[1,8400,84]` | 19.83 MiB |
| `fotonetl-p2` | 5,326,816 | 5,279,208 | 8.116G | 16.232G | 101 | `[1,34000,84]` | 20.42 MiB |
| `fotonetx` | 8,689,998 | 8,644,259 | 10.819G | 21.638G | 90 | `[1,8400,84]` | 33.27 MiB |
| `fotonetx-p2` | 8,920,592 | 8,869,188 | 13.079G | 26.159G | 107 | `[1,34000,84]` | 34.16 MiB |

## RTX 4060 eager FP32 forward speed

FPS is batch size divided by arithmetic-mean synchronized forward latency.
The latency columns are per-forward p50/p95. No preprocessing, postprocessing,
data loading, AMP, or compilation is included.

| Model | B1 p50/p95 | B1 FPS | B8 p50/p95 | B8 FPS | Peak B1/B8 VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fotonetn` | 4.225/6.395 ms | 218.33 | 11.050/11.385 ms | 723.43 | 30.44/218.08 MiB |
| `fotonetn-p2` | 6.338/8.606 ms | 154.31 | 28.199/28.854 ms | 281.91 | 93.88/730.20 MiB |
| `fotonets` | 4.664/8.331 ms | 191.57 | 17.402/17.718 ms | 460.46 | 40.54/275.94 MiB |
| `fotonets-p2` | 6.306/8.809 ms | 152.78 | 33.229/34.030 ms | 240.25 | 99.48/755.17 MiB |
| `fotonetm` | 5.431/7.900 ms | 174.89 | 23.131/23.662 ms | 346.32 | 50.66/308.40 MiB |
| `fotonetm-p2` | 6.846/9.856 ms | 139.88 | 42.039/42.856 ms | 189.77 | 110.33/800.41 MiB |
| `fotonetl` | 5.388/8.902 ms | 169.75 | 34.934/35.414 ms | 228.88 | 67.69/393.79 MiB |
| `fotonetl-p2` | 7.051/8.898 ms | 137.33 | 57.251/57.898 ms | 139.11 | 124.80/852.15 MiB |
| `fotonetx` | 6.581/7.833 ms | 144.01 | 50.672/51.304 ms | 157.77 | 91.71/485.24 MiB |
| `fotonetx-p2` | 9.046/11.530 ms | 106.40 | 77.886/81.083 ms | 102.17 | 144.70/914.32 MiB |

Hardware/software: RTX 4060 8GB, Windows build 26100, Python 3.10.11,
PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, TF32 disabled. Each speed result
used 30 warmups and 100 timed forwards; VRAM used a separate post-warmup peak.
THOP 0.1.1 measured MACs and does not charge unsupported elementwise/grid ops.

## Planned scale resizing

This is a roadmap, not the current graph and not a performance claim.

| Scale | Planned center | Allowed band |
| --- | ---: | ---: |
| S | 2.20M params | 2.02M–2.38M |
| M | 5.00M params | 4.50M–5.50M |
| L | 11.40M params | 10.40M–12.40M |
| X | 33.80M params | 28.80M–38.80M |

Channels, depth, MACs, FLOPs, memory, and latency will change when those new
graphs are designed. Values will be published only after direct measurement.
