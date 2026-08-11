# Inference and results

Official weights are still training. Until their verified release, pass a
supported checkpoint path; named models construct untrained graphs.

```python
from fotonet import Fotonet

model = Fotonet("path/to/checkpoint.pt")
results = model.predict("image.jpg", conf=0.25, imgsz=640)
result = results[0]
```

Folders and batches:

```python
for result in model.predict("images/", batch=8):
    print(len(result), result.orig_shape)
```

OpenCV BGR frames:

```python
result = model.predict_bgr(frame, conf=0.25)[0]
```

Image, array, and tensor prediction returns `list[Results]`, one item per
input. Video/webcam calls return an iterator only with `stream=True`.

## Results

```python
for detection in result.boxes:
    print(detection.cls_id, detection.cls, detection.conf)
    print(detection.xywh, detection.xyxy)

result.save("prediction.jpg")
json_text = result.to_json()
```

`Results`, `DetectionBox`, `DetectionBoxes`, and transform types are exported
from `fotonet`. Tensor input accepts RGB `[3,H,W]` or `[B,3,H,W]` values in
`[0,1]` or `[0,255]`. Tensor calls avoid retaining host images by default; use
`retain_images=True` for plotting, saving, or crops.

## Video tracking

```python
for result in model.track("video.mp4", stream=True, persist=True):
    print(result.to_json())
```

The built-in tracker is same-class IoU association, not motion estimation or
re-identification.

## Runtime measurements

Measured eager FP32 forward results and their exact hardware/methodology are in
[Models and measured runtime](model-zoo.md). They exclude decoding,
preprocessing, postprocessing, and I/O.
