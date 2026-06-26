# Training

This alpha release is not recommended for production workflows. The training path exists so users can experiment, reproduce runs, and help harden the project.

## Dataset Format

FOTO-NET expects YOLO-format labels:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the image size.

## `data.yaml`

A minimal dataset config should include:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 80
names:
  0: class_0
```

Use paths that are valid on your machine. Do not commit local dataset paths to shared configs.

## Python Training

```python
from fotonet import FOTONET

model = FOTONET("fotonetn")
model.train(data="data.yaml", epochs=100, imgsz=640, batch=16)
```

## CLI Training

```bash
fotonet train model=fotonetn data=data.yaml epochs=100 imgsz=640 batch=16
```

## Resume and Pretrained Modes

`resume=True` means continue from a full training checkpoint such as `fotonet_last.pt`. It restores training state, optimizer state, scheduler state, scaler state, and epoch position.

`pretrained=True` means load model weights only and start with a fresh optimizer and scheduler.

`resume=True` and `pretrained=True` are mutually exclusive.

`fotonet_best.pt` can be a slim inference checkpoint. `fotonet_last.pt` is the resumable training checkpoint.
