# Training and resume

fotonet supports one production training policy: ordinary uniform sampling.
The deploy graph uses one-to-one predictions; the one-to-many branch is
training-only and can be stripped from inference checkpoints.

## Dataset YAML

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 2
names:
  0: cat
  1: dog
```

Labels use normalized YOLO rows:

```text
class_id x_center y_center width height
```

The default `annotation_policy="fix"` clamps edge-crossing boxes and drops
malformed/duplicate rows. Use `annotation_policy="error"` for a strict audit.
Public validation requires an independent `val` source and never silently
uses training data.

## CLI launcher

The CLI uses `key=value` arguments and validates them before starting a long run.

```bash
python -m fotonet.cli.main train \
  model=fotonete \
  data=path/to/data.yaml \
  epochs=300 \
  batch=16 \
  imgsz=640 \
  run_dir=runs/fotonete \
  dry_run=true
```

Inspect its resolved graph/checkpoint identity and arguments without starting
training:

```bash
fotonet train model=fotonete data=path/to/data.yaml dry_run=true
```

## Resume after interruption

```bash
fotonet train \
  model=fotonete \
  data=path/to/data.yaml \
  epochs=300 \
  batch=16 \
  run_dir=runs/fotonete \
  resume=runs/fotonete/fotonet_last.pt
```

Resume restores model, EMA, optimizer, scheduler, AMP scaler, epoch/global
step, and RNG state. It rejects a different model, class count/order, training
protocol, or incomplete checkpoint.

## Python API

```python
from fotonet import Fotonet

model = Fotonet("fotonete", nc=2)
summary = model.train(
    data="data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    save_dir="runs/fotonete",
)
```

Resume through Python:

```python
model = Fotonet("runs/fotonete/fotonet_last.pt")
summary = model.train(
    data="data.yaml",
    epochs=300,
    weights="runs/fotonete/fotonet_last.pt",
    resume=True,
    save_dir="runs/fotonete",
)
```

`resume=True` continues complete state. `pretrained=True` starts a fresh
optimizer/scheduler from a supported weight. They are mutually exclusive.

Checkpoint deserialization is always tensor-only. There is no unsafe pickle
switch and no metadata-free reconstruction.

## Validation semantics

`val_conf=0.0` retains finite detections for score-ranked COCO evaluation.
`operating_conf` and `operating_iou` report a separate deployment operating
point. Sanitized/noncanonical annotations are labeled as such and must not be
reported as official COCO metrics.
