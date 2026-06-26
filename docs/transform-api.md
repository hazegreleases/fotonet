# Transform API

The transform API is a mutable spatial helper attached to each detection box.

Read `box.xywh` for the original detection center and size. Use `box.transform` for working spatial operations such as cropping, moving, expanding, anchoring, and containment checks.

## Basic Usage

```python
from fotonet import AnchorPoint

box = results.boxes[0]
crop = (
    box.transform
    .setAnchor(AnchorPoint.CENTER)
    .pixelExpand(40)
    .clamp()
    .crop(results.orig_img)
)
```

## Properties

- `box.transform.xywh`: normalized center x, center y, width, height
- `box.transform.xyxy`: normalized x1, y1, x2, y2
- `position`: normalized anchor position
- `pixelPosition`: anchor position in pixels
- `size`: normalized width and height
- `pixelSize`: width and height in pixels
- `corner[index]`: normalized corner point, 1 through 4
- `side[index]`: normalized side midpoint, 1 through 4

## Methods

- `setAnchor(anchor)`: set the anchor used by scale and expansion operations
- `move((dx, dy))`: move in normalized units
- `pixelMove((dx, dy))`: move in pixels
- `scale((sx, sy))`: scale about the active anchor
- `expand(padding)`: expand in normalized units
- `pixelExpand(padding)`: expand in pixels
- `clamp()`: clamp the box to image bounds
- `setBox(mode=1)`: make the region square
- `setAspectRatio((w, h), mode=1)`: enforce an aspect ratio
- `crop(image)`: crop from a PIL image or NumPy array
- `contains(point=...)`: normalized point or region containment
- `pixelContains(point=...)`: pixel point or region containment
- `iou(other)`: intersection over union
- `distance(point)`: normalized distance from active anchor
- `focus(x, y, w, h)`: focus on a relative subregion
- `focusReset()`: restore the original detection box

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
