# Transform API

The transform API is a mutable spatial helper attached to each detection box.

Read `box.xywh` for the original detection center and size. Use `box.transform` for working spatial operations such as cropping, moving, expanding, anchoring, and containment checks.

## Basic Usage

```python
from fotonet import AnchorPoint

# `predict()` returns a list: choose the image result first.
result = results[0]
box = result.boxes[0]
crop = (
    box.transform
    .set_anchor(AnchorPoint.CENTER)
    .pixel_expand(40)
    .clamp()
    .crop(result.orig_img)
)
```

## Properties

- `box.transform.xywh`: normalized center x, center y, width, height
- `box.transform.xyxy`: normalized x1, y1, x2, y2
- `position`: normalized anchor position
- `pixel_position`: anchor position in pixels
- `size`: normalized width and height
- `pixel_size`: width and height in pixels
- `corner[index]`: normalized corner point, 1 through 4
- `side[index]`: normalized side midpoint, 1 through 4

## Methods

- `set_anchor(anchor)`: set the anchor used by scale and expansion operations
- `move((dx, dy))`: move in normalized units
- `pixel_move((dx, dy))`: move in pixels
- `scale((sx, sy))`: scale about the active anchor
- `expand(padding)`: expand in normalized units
- `pixel_expand(padding)`: expand in pixels
- `clamp()`: clamp the box to image bounds
- `set_box(mode=1)`: make the region square
- `set_aspect_ratio((w, h), mode=1)`: enforce an aspect ratio
- `crop(image)`: crop from a PIL image or NumPy array
- `contains(point=...)`: normalized point or region containment
- `pixel_contains(point=...)`: pixel point or region containment
- `iou(other)`: intersection over union
- `distance(point)`: normalized distance from active anchor
- `focus(x, y, w, h)`: focus on a relative subregion
- `focus_reset()`: restore the original detection box

## Anchors

Available anchors include:

- `AnchorPoint.CENTER`
- `AnchorPoint.TOP_LEFT`
- `AnchorPoint.TOP_RIGHT`
- `AnchorPoint.BOTTOM_LEFT`
- `AnchorPoint.BOTTOM_RIGHT`
- `AnchorPoint.TOP`
- `AnchorPoint.BOTTOM`
- `AnchorPoint.LEFT`
- `AnchorPoint.RIGHT`
