# Quick Start

This alpha release is a demo of what is coming. It is not recommended for production workflows.

## Construct a Model

```python
from fotonet import FOTONET

model = FOTONET("fotonetn")
```

Constructing a named scale builds the architecture. Released weights are needed for meaningful detections.

## Run Inference With a Checkpoint

```python
from fotonet import FOTONET

model = FOTONET("fotonet-n")
results = model.predict("image.jpg", conf=0.25)
print(results)
```

## Inspect Results

```python
for box in results.boxes:
    print(box.cls, box.conf, box.xywh, box.xyxy)
```

## CLI Inference

```bash
python -m fotonet.cli.main predict model=fotonet-n source=image.jpg conf=0.25 save=true
```

## Train a YOLO Dataset

```bash
fotonet train model=fotonetn data=data.yaml epochs=100 imgsz=640 batch=16
```
