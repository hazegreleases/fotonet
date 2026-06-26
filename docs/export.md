# Export

FOTO-NET alpha provides export helpers for inference graphs. Export support depends on the local toolchain.

## ONNX

```python
from fotonet import FOTONET

model = FOTONET("fotonet-n")
model.export(format="onnx", path="dev-tools/runs/fotonet.onnx", imgsz=640)
```

ONNX export writes a metadata JSON file next to the artifact.

## TorchScript

```python
model.export(format="torchscript", path="fotonet.torchscript", imgsz=640)
```

## TensorRT

TensorRT export requires `trtexec` in `PATH`.

```python
model.export(format="tensorrt", path="fotonet.engine", imgsz=640, half=True)
```

## CoreML

CoreML code exists, but this alpha does not treat CoreML as certified unless a fresh platform-specific verification is published.

## Output Format

Exported graphs output the raw one-to-one inference tensor. Application code is responsible for postprocessing unless a future release adds packaged postprocess graphs.
