# Export

fotonet provides versioned export helpers. Export support depends on the local toolchain.

## ONNX

```python
from fotonet import Fotonet

model = Fotonet("my_checkpoint.pt")
artifact = model.export(format="onnx", path="exports/fotonet.onnx", imgsz=640)
print(artifact["artifact"], artifact["metadata"])
```

ONNX export writes a metadata JSON file next to the artifact and, by default, checks native-vs-ONNX Runtime numerical parity before returning. The metadata specifies RGB NCHW `[0,1]` input, class names, raw output layout, and the stride-padding coordinate contract.

Use dynamic H/W ONNX when deployment requires variable input sizes:

```python
model.export(format="onnx", path="exports/fotonet_dynamic.onnx", imgsz=(640, 960), dynamic=True)
```

The graph pads right/bottom to its maximum stride internally and maps normalized boxes back to the unpadded caller input. Static artifacts accept only their recorded input shape.

## TorchScript

```python
model.export(format="torchscript", path="fotonet.torchscript", imgsz=640)
```

TorchScript archives embed the same FOTO-NET metadata contract as the sidecar,
so copying the artifact alone preserves its class count and preprocessing
rules. The default `dynamic=False` keeps the public runtime on the recorded
input size. Set `dynamic=True` to validate and allow variable batch/H/W input
through `Fotonet("fotonet.torchscript").predict(...)`:

```python
model.export(
    format="torchscript",
    path="fotonet_dynamic.torchscript",
    imgsz=(640, 960),
    dynamic=True,
)
```

## TensorRT

TensorRT export requires `trtexec` in `PATH`.

```python
model.export(format="tensorrt", path="fotonet.engine", imgsz=640, half=True)
```

## CoreML

CoreML code exists, but this alpha does not treat CoreML as certified unless a fresh platform-specific verification is published.

## Output Format

Exported graphs output `[B, N, nc + 4]`: class logits followed by normalized
`xywh`. Application code applies sigmoid, class selection, confidence
thresholding, clipping, and a top-k cap. `Fotonet("artifact.onnx").predict(...)`
provides that postprocess for ONNX/TorchScript artifacts. Export metadata must
declare `export_schema: 1` and the exact architecture identity.

## INT8 ONNX

INT8 export is calibrated static-QDQ ONNX, not a fake cast. Supply representative preprocessed RGB NCHW batches in `[0,1]`:

```python
artifact = model.export(
    format="onnx",
    path="exports/fotonet_int8.onnx",
    imgsz=640,
    int8=True,
    calibration_data=calibration_batches,
)
```

TensorRT engine building validates the build command and shape profile; perform platform-specific numerical/latency validation before production deployment.
