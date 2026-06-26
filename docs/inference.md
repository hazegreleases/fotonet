# Inference

## Python Inference

```python
from fotonet import FOTONET

model = FOTONET("fotonet-n")
results = model.predict("image.jpg", conf=0.25, imgsz=640)
```

## Folder Inference

```python
from fotonet import FOTONET

model = FOTONET("fotonet-n")
results = model.predict("images", batch=8)
for result in results:
    print(result)
```

## BGR Frames

OpenCV users can call `predict_bgr()` with BGR `uint8` frames:

```python
result = model.predict_bgr(frame, conf=0.25)
```

## Results Objects

`Results` contains:

- `orig_img`: original image object or array
- `boxes`: iterable detection collection
- `scores`: confidence tensor
- `classes`: class id tensor
- `names`: class name mapping

Each `DetectionBox` exposes:

- `idx`
- `conf`
- `cls_id`
- `cls`
- `xywh`
- `xyxy`
- `transform`

## Visualization and Export Helpers

```python
plot = results.plot()
results.save("dev-tools/runs/prediction.jpg")
json_text = results.to_json()
```

`to_pandas()` requires pandas.

## Inference Timing

Inference times are not out yet because the benchmark computer is IO bottlenecked, so the timing numbers are not reliable enough to publish.
