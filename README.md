# fotonet

fotonet is an open-source, compact NMS-free object detector for computer vision
and machine learning workflows. It is built with PyTorch and provides a Python
API, resumable training, COCO-style validation, small-object P2 variants, and
ONNX or TorchScript export.

The repository is being published while the first official Nano checkpoint is
still training. Model names currently construct untrained architectures. The
trained weight, SHA256 checksum, canonical validation report, and automatic
download hook will be published after training and release verification finish.
There is no public AP claim yet.

Documentation: https://hazegreleases.github.io/fotonet/

Current source/package version: `v0.8.0b2` (beta).

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

## Inference

```python
from fotonet import Fotonet

model = Fotonet("path/to/checkpoint.pt")
results = model.predict("image.jpg", conf=0.25, imgsz=640)

for detection in results[0].boxes:
    print(detection.cls, detection.conf, detection.xyxy)
```

Image and tensor calls return `list[Results]`. Video/webcam prediction returns
an iterator only when `stream=True`.

## Box transforms

```python
from fotonet import AnchorPoint

crop = (
    results[0].boxes[0].transform
    .set_anchor(AnchorPoint.CENTER)
    .pixel_expand(40)
    .clamp()
    .crop(results[0].orig_img)
)
```

## Training

Download the public launcher, then point it at a YOLO-format dataset:

```bash
curl -L https://hazegreleases.github.io/fotonet/examples/train.py -o train.py
```

```bash
python train.py \
  --model fotonetn \
  --data path/to/data.yaml \
  --epochs 300 \
  --batch 16 \
  --run-dir runs/fotonetn
```

Resume an interrupted run without starting a second training protocol:

```bash
python train.py \
  --model fotonetn \
  --data path/to/data.yaml \
  --epochs 300 \
  --batch 16 \
  --run-dir runs/fotonetn \
  --resume
```

No training is started by importing the package or by a launcher `--dry-run`.

## Export

```python
from fotonet import Fotonet

model = Fotonet("path/to/checkpoint.pt")
output = model.export(format="onnx", path="exports/fotonet.onnx", imgsz=640)
print(output["artifact"], output["metadata"])
```

Supported checkpoints and exports are self-identifying and tensor-only loaded.
Missing or unknown schema versions fail closed; the runtime does not infer a
model from filenames or tensor shapes.

## Architecture and measurements

Every public model uses the same production `Backbone`, `Neck`, `Head`, and
`Detector` implementation under `fotonet.models.v1`. P2 variants add a
stride-4 prediction level for small-object experiments.

The current Nano graph has 1,042,936 training parameters (1,005,932 fused
deployment parameters), 1.313 GMAC / 2.626 GFLOP at 640x640, and measured
218.33 images/s at batch 1 or 723.43 images/s at batch 8 on the declared RTX
4060 FP32 benchmark. These are graph/runtime measurements, not accuracy claims.
See [the model table](docs/model-zoo.md) for all ten variants and methodology.

S, M, L, and X will be resized in a later architecture revision. Planned
parameter centers are 2.2M, 5.0M, 11.4M, and 33.8M respectively, with the
explicit bands documented in the model table. Their future MAC/FLOP values
will be measured after the graphs exist; they are not estimated here.

## Documentation

- [Installation](docs/installation.md)
- [Quick start](docs/quickstart.md)
- [Inference and results](docs/inference.md)
- [Training and resume](docs/training.md)
- [Models and measured runtime](docs/model-zoo.md)
- [Model configuration](docs/model-config.md)
- [Export](docs/export.md)
- [Transform API](docs/transform-api.md)
- [Security](docs/security.md)
- [Contributing](docs/contributing.md)
- [Production cleanup boundary](docs/system-cleanup.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
