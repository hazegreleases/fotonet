# Quick Start

`Fotonet("fotonete")` loads the verified `fotonete.pt` checkpoint from the
public `v1.0.0` GitHub release. The checkpoint is downloaded once, verified by
SHA-256, and cached locally; it is not bundled in the Python package.

## Construct a Model

```python
from fotonet import Fotonet

model = Fotonet("fotonete")
```

Use `Fotonet()` when you intentionally need an untrained architecture for
training or custom checkpoint loading.

To use a local checkpoint explicitly, pass its path:

```python
from fotonet import Fotonet

model = Fotonet("path/to/checkpoint.pt")
```

Set `FOTONETE_MODEL_PATH` to override the release checkpoint, or set
`FOTONETE_CACHE_DIR` to choose the download cache directory.

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
python -m fotonet.cli.main train model=fotonete data=data.yaml epochs=100 imgsz=640 batch=16 run_dir=runs/fotonete
```

Add `dry_run=true` to resolve the graph, checkpoint, paths, and training arguments
without constructing a trainer or starting training. Add `resume=<checkpoint>` to continue
the same run from `runs/fotonete/fotonet_last.pt`.
