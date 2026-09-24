# FOTONET Documentation

`fotonet` is a Python object-detection library for running models, training on
your own data, evaluating detections, and exporting a model for deployment.

## Start here

- [Quickstart](quickstart.md) — install the package and run one image.
- [Installation](installation.md) — install CPU or CUDA dependencies.
- [Inference](inference.md) — use images, video, and Python frames.
- [Training and resume](training.md) — train and continue a run.
- [Models and runtime](model-zoo.md) — check the supported model and output format.
- [Model configuration](model-config.md) — understand the fixed model settings.
- [Export](export.md) — create ONNX, TensorRT, CoreML, or TorchScript artifacts.

## Install

```bash
python -m pip install fotonet
```

## Run inference

Download a checkpoint from the GitHub Releases page, then pass its path to
`Fotonet`:

```python
from fotonet import Fotonet

model = Fotonet("path/to/fotonet_last.pt")
results = model.predict("image.jpg", conf=0.25, imgsz=640)

for detection in results[0].boxes:
    print(detection.cls, detection.conf, detection.xyxy)
```

## Train a model

Prepare a YAML dataset configuration and start a run:

```bash
fotonet train \
  model=fotonete \
  data=path/to/data.yaml \
  epochs=300 \
  batch=16 \
  imgsz=640 \
  run_dir=runs/fotonete
```

Use `resume=runs/fotonete/fotonet_last.pt` to continue an interrupted run.
Training outputs and model weights stay outside the Git source tree.

## Checkpoints and measurements

The supported model ID is `fotonete`. Model weights are distributed through
GitHub Releases and are not committed to Git. Benchmark figures on the website
are measured on a stated local test machine; they are not guarantees for every
device or workload.

## License

Apache License 2.0. See [LICENSE](../LICENSE).