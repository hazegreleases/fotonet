# fotonet

`fotonet` is a compact Python object-detection library for local inference,
training, evaluation, and model export. The supported model is `fotonete`.

## Install

```bash
python -m pip install fotonet
```

For a development checkout:

```bash
git clone https://github.com/hazegreleases/fotonet.git
cd fotonet
python -m pip install -e ".[dev]"
```

## Quick start

Download a release checkpoint, then run inference from Python:

```python
from fotonet import Fotonet

model = Fotonet("path/to/fotonet_last.pt")
results = model.predict("image.jpg", conf=0.25, imgsz=640)

for detection in results[0].boxes:
    print(detection.cls, detection.conf, detection.xyxy)
```

For a BGR frame from OpenCV:

```python
frame = cv2.imread("image.jpg")
results = model.predict_bgr(frame, conf=0.25, imgsz=640)
```

## Train

Training uses a YAML dataset configuration and writes checkpoints to a local run directory:

```bash
fotonet train \
  model=fotonete \
  data=path/to/data.yaml \
  epochs=300 \
  batch=16 \
  imgsz=640 \
  run_dir=runs/fotonete
```

Resume an interrupted run:

```bash
fotonet train \
  model=fotonete \
  data=path/to/data.yaml \
  resume=runs/fotonete/fotonet_last.pt
```

Weights and training outputs are not stored in the Git repository.

## Export

```python
from fotonet import Fotonet

model = Fotonet("path/to/fotonet_last.pt")
output = model.export(
    format="onnx",
    path="exports/fotonete.onnx",
    imgsz=640,
)
print(output["artifact"])
```

See the [export guide](docs/export.md) for available formats and optional dependencies.

## Documentation

- [Documentation portal](docs/index.md)
- [Installation](docs/installation.md)
- [Quick start](docs/quickstart.md)
- [Inference and results](docs/inference.md)
- [Training and resume](docs/training.md)
- [Models and runtime](docs/model-zoo.md)
- [Model configuration](docs/model-config.md)
- [Export](docs/export.md)
- [Transform API](docs/transform-api.md)
- [Security](docs/security.md)
- [Contributing](docs/contributing.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).