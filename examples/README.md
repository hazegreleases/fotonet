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

This example creates its own NumPy image and needs no checkpoint:

```bash
python examples/transform_crop.py
```
