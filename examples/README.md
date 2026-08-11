# Runnable examples

Run these commands from the repository root after installing the package.
Inference and export examples require a self-identifying checkpoint.

## One image with an interactive picker

```bash
python examples/predict_image.py --model weights/fotonetn.pt --source image.jpg --save runs/example
```

Omit `--source` to open the system file picker. In the preview window, press
`r` to choose another image or `q`/Escape to exit.

## A folder of images

```bash
python examples/predict_folder.py --model weights/fotonetn.pt --source images --batch 8 --save-dir runs/folder
```

## Training launcher

The example delegates to the repository's locked production launcher; it does
not define a second training policy.

```bash
python examples/train_custom_yolo.py --model fotonetn --data data.yaml --epochs 100 --batch 16 --run-dir runs/custom
```

Validate every path and identity without starting training:

```bash
python examples/train_custom_yolo.py --model fotonetn --data data.yaml --run-dir runs/custom --dry-run
```

Resume the same run:

```bash
python examples/train_custom_yolo.py --model fotonetn --data data.yaml --epochs 100 --run-dir runs/custom --resume
```

## ONNX export

```bash
python examples/export_onnx.py --model weights/fotonetn.pt --path exports/fotonetn.onnx --imgsz 640
```

## Box transforms

The deterministic smoke example creates its own NumPy image and needs no
checkpoint:

```bash
python examples/transform_crop.py
```

Crop a stored normalized region from a real image without running a model:

```bash
python examples/transform_region.py image.jpg --xywh .52 .48 .30 .62 --aspect 4:5 --anchor bottom --padding 28 --output outputs/card.jpg
```

Create confidence-ordered person crops and a JSON manifest:

```bash
python examples/extract_detection_crops.py weights/fotonetn.pt street.jpg --class-name person --aspect 4:5 --anchor bottom --padding 24 --output-dir outputs/people
```

Use `--focus 0 0 1 .55` with that script to crop the upper 55% of each
detection before aspect fitting and padding.

Test bottom contact points against a source-pixel zone, then write an annotated
image and JSON decisions:

```bash
python examples/anchor_zone_filter.py weights/fotonetn.pt entrance.jpg --zone 180 240 940 700 --anchor bottom --class-name person --output outputs/entrance-zone.jpg
```

The full spatial semantics, containment policies, anchor behavior, and safe
operation order are documented in `docs/transform-api.md`.

For ordered video, log tracked enter/exit transitions as JSON Lines:

```bash
python examples/track_zone_events.py weights/fotonetn.pt entrance.mp4 --zone 180 240 940 700 --anchor bottom --class-name person --output outputs/zone-events.jsonl
```

This uses the built-in class-aware IoU tracker. It records visible anchor
transitions; it is not motion estimation, re-identification, or segmentation.
