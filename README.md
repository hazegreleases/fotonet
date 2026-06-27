![FOTO-NET alpha banner](https://raw.githubusercontent.com/hazegreleases/boringstuffreally/main/Alpha%20banner.png)

# FOTO-NET

FOTO-NET is a lightweight NMS-free object detector with an Ultralytics-like Python API and an application-friendly transform layer for working with detection boxes.

This is the alpha release, everything is subject to change. Not recommended for production workflows, just a demo of what is coming.

## What It Is

FOTO-NET focuses on practical object detection:

- compact model scales
- NMS-free one-to-one inference
- optional one-to-many training supervision
- small-object friendly P2 variants
- direct Python results objects
- transform helpers for crops, anchors, pixel movement, containment, and box manipulation
- export paths for ONNX, TorchScript, TensorRT, and CoreML where dependencies are available

## Alpha Status

The alpha release is intended for experimentation, training runs, integration tests, and feedback. The core Python API, training path, inference path, transform API, and ONNX export should be usable. CoreML export code exists, but it should not be treated as certified until fresh platform-specific verification is published.

## Install

```bash
pip install fotonet
```

### Or

```bash
git clone https://github.com/hazegreleases/fotonet.git
cd fotonet
python -m pip install torch torchvision numpy pillow pyyaml opencv-python matplotlib scipy tqdm
```

## Quick Start

```python
from fotonet import FOTONET

model = FOTONET("fotonetn")
results = model.predict("image.jpg", conf=0.25)

for box in results.boxes:
    print(box.cls, box.conf, box.xyxy)
```

CLI:

```bash
python -m fotonet.cli.main predict model=fotonet-n source=image.jpg conf=0.25 save=true
```

## Transform API

```python
from fotonet import AnchorPoint

box = results.boxes[0]
crop = (
    box.transform
    .setAnchor(AnchorPoint.CENTER)
    .pixelExpand(40)
    .clamp()
    .crop(results.orig_img)
)
```

The raw box formats remain available through `box.xywh`, `box.xyxy`, and `results.boxes.numpy()`.

## Training

FOTO-NET expects YOLO-format labels:

```text
class_id x_center y_center width height
```

Train through Python:

```python
from fotonet import FOTONET

model = FOTONET("fotonetn")
model.train(data="data.yaml", epochs=100, imgsz=640, batch=16)
```

Train through CLI:

```bash
fotonet train model=fotonetn data=data.yaml epochs=100 imgsz=640 batch=16
```

## Export

```python
from fotonet import FOTONET

model = FOTONET("fotonet-n")
model.export(format="onnx", path="dev-tools/runs/fotonet.onnx", imgsz=640)
```

## Model Weights

The alpha repository includes `fotonet-n` as the public nano checkpoint. Training outputs and development weights live outside the public surface.

## Alpha Checkpoint Metrics

`fotonet-n` is the alpha nano checkpoint included in this repository.

| Checkpoint | mAP@.50:.95 | Parameters | MACs at 640 | GFLOPs at 640 |
|---|---:|---:|---:|---:|
| `fotonet-n` | 22.68% | 2.75M | 2.43G | 4.85G |

Inference times are not out yet because the benchmark computer is IO bottlenecked, so the timing numbers are not reliable enough to publish.

## Documentation

- Installation: `docs/installation.md`
- Quick start: `docs/quickstart.md`
- Inference: `docs/inference.md`
- Training: `docs/training.md`
- Export: `docs/export.md`
- Transform API: `docs/transform-api.md`
- Model zoo: `docs/model-zoo.md`

## License

FOTO-NET is licensed under the Apache License, Version 2.0.
