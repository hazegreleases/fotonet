# Installation

## Requirements

FOTO-NET alpha targets Python 3.10 or newer. A CUDA-capable PyTorch install is recommended for training, but CPU installs are useful for API tests, documentation examples, and small export smoke checks.

Install PyTorch from the official PyTorch instructions for your platform first when you need a specific CUDA build.

## Public Alpha Checkout

```bash
git clone https://github.com/hazegreleases/fotonet.git
cd fotonet
python -m pip install torch torchvision numpy pillow pyyaml opencv-python matplotlib scipy tqdm
```

Run Python examples from the repository root so the local `fotonet` package is importable.

```bash
python examples/transform_crop.py
```

## Optional ONNX Dependencies

```bash
python -m pip install onnx onnxsim
```

ONNX export works best when `onnx` is installed. `onnxsim` is optional and only used for graph simplification.

## Optional Metrics Dependencies

```bash
python -m pip install pycocotools
```

`pycocotools` enables COCO-style validation metrics. Without it, FOTO-NET can still run model construction, inference, training code paths, and local tests that do not require COCO evaluation.

## CUDA Notes

Use a PyTorch build that matches your driver and CUDA runtime. Training speed depends heavily on GPU memory, batch size, image size, and whether validation is running on the full dataset.

## Troubleshooting

If imports fail after cloning, run:

```bash
python -c "from fotonet import FOTONET; print(FOTONET('fotonetn').nc)"
```

If export dependencies are missing, install them directly:

```bash
python -m pip install onnx onnxsim
```
