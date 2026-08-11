# Quick Start

The first official weight is currently training. Until its verified release,
named models construct untrained graphs and inference requires your supported checkpoint.

## Construct a Model

```python
from fotonet import Fotonet

model = Fotonet("fotonetn")
```

Constructing a named scale builds an untrained architecture. Load a trusted checkpoint before expecting meaningful detections.

## Run Inference With a Checkpoint

```python
from fotonet import Fotonet

model = Fotonet("my_checkpoint.pt")
results = model.predict("image.jpg", conf=0.25)
result = results[0]
print(result)
```

## Inspect Results

```python
for box in result.boxes:
    print(box.cls, box.conf, box.xywh, box.xyxy)
```

## CLI inference

```bash
fotonet predict model=my_checkpoint.pt source=image.jpg conf=0.25 save=true
```

## Train a YOLO dataset

```bash
curl -L https://hazegreleases.github.io/fotonet/examples/train.py -o train.py
python train.py --model fotonetn --data data.yaml --epochs 100 --imgsz 640 --batch 16 --run-dir runs/fotonetn
```

Add `--dry-run` to resolve the graph, checkpoint, paths, and training arguments
without constructing a trainer or starting training. Add `--resume` to continue
the same run from `runs/fotonetn/fotonet_last.pt`.
