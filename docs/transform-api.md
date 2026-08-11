# AnchorPoint and transform API

The transform layer turns a detection rectangle into application geometry:
crops, fixed-aspect cards, relative subregions, contact-point zone tests, and
spatial relationships. It does not modify model output or confidence.

## Mental model

`DetectionBox.xywh` is the original normalized detection. `box.transform` is a
mutable working region initialized from that detection. Its coordinates use
normalized `0..1` image space; pixel-prefixed properties and methods use the
source image dimensions.

```python
result = model.predict("image.jpg", retain_images=True)[0]
box = result.boxes[0]

print(box.xywh)                  # original normalized detection
print(box.transform.xyxy)        # active normalized working region
print(tuple(box.transform.pixel_size))
```

Most mutating methods return the transform and can be chained. Calling
`transform.focus(x, y, w, h)` returns the applied `FocusRegion`, so apply focus
on one line and continue the transform chain on the next.

## Anchor behavior

An anchor is the point that remains fixed during `scale`, `expand`, and
`pixel_expand`:

- `CENTER`: context grows on every side.
- `TOP`: the top midpoint stays fixed; context grows left, right, and down.
- `BOTTOM`: the bottom midpoint stays fixed; context grows left, right, and up.
- `LEFT` / `RIGHT`: the selected side midpoint stays fixed.
- Corner anchors grow toward the opposite quadrant.

This makes `BOTTOM` useful for people or vehicles whose floor-contact point
should not move when context is added.

```python
from fotonet import AnchorPoint

region = box.transform
region.set_anchor(AnchorPoint.BOTTOM).pixel_expand(24)
fixed = region.pixel_position
region.set_aspect_ratio((4, 5), mode=1)  # centered, expand to preserve content
shifted = region.pixel_position
region.pixel_move((fixed.x - shifted.x, fixed.y - shifted.y)).clamp()
crop = region.crop(result.orig_img)
```

`set_box` and `set_aspect_ratio` fit around the current center. Record and
restore the pixel anchor as above when exact aspect and a non-center physical
anchor both matter. Mode `1` expands to preserve the region; mode `0` shrinks
and may cut it. Clamping at an image boundary may tighten the final ratio.

## Focus on a subregion

`FocusRegion(x, y, w, h)` is relative to the active region. `(0, 0)` is its
top-left; `(1, 1)` reaches its bottom-right.

```python
from fotonet import FocusRegion

upper_body = FocusRegion(0.0, 0.0, 1.0, 0.55)
region = box.transform
region.focus = upper_body
region.pixel_expand(12).clamp()
crop = region.crop(result.orig_img)
region.focus_reset()
```

`FocusRegion.blend(other, alpha)` interpolates application presets. Dividing
one focus region by another is a 50/50 blend. Repeated focus operations are
relative to the current region; use `focus_reset()` before starting a new path.

## Zone gating with a contact anchor

Build a zone as another `BoxTransform`, select the representative detection
anchor, then test that pixel point:

```python
from fotonet import AnchorPoint, BoxTransform

image_w, image_h = result.image_size
x1, y1, x2, y2 = 180, 240, 940, 700
zone = BoxTransform(
    ((x1 + x2) / (2 * image_w),
     (y1 + y2) / (2 * image_h),
     (x2 - x1) / image_w,
     (y2 - y1) / image_h),
    image_size=result.image_size,
).clamp()

for box in result.boxes:
    foot = box.transform.set_anchor(AnchorPoint.BOTTOM).pixel_position
    if zone.pixel_contains(point=foot):
        print(box.cls, box.conf, tuple(foot))
```

Point containment and area containment answer different questions:

```python
# Is the contact point inside the zone?
inside_by_contact = zone.pixel_contains(point=foot)

# Is at least 75% of this detection covered by the zone rectangle?
mostly_inside = box.transform.pixel_contains(
    x=(zone_x1, zone_x2),
    y=(zone_y1, zone_y2),
    threshold=0.75,
)
```

## Properties

- `xywh`, `xyxy`: active normalized geometry
- `position`, `pixel_position`: active anchor position
- `size`, `pixel_size`: active width and height
- `corner[1..4]`: top-left, top-right, bottom-right, bottom-left
- `side[1..4]`: top, right, bottom, left midpoints
- `pixel_corner`, `pixel_side`: pixel-space forms with the same indices

Point and size properties return `Vector2`, supporting `.x`, `.y`, unpacking,
and indices `0`/`1`. `BoxTransform(..., image_size=(width, height))` always uses
width before height.

## Methods

- `set_anchor(anchor)`
- `move((dx, dy))`, `pixel_move((dx, dy))`
- `scale((sx, sy))`
- `expand(padding)`, `pixel_expand(padding)`
- `clamp()`
- `set_box(mode=1)`, `set_aspect_ratio((w, h), mode=1)`
- `crop(image)`
- `contains(...)`, `pixel_contains(...)`
- `overlaps(other)`, `iou(other)`
- `distance(point)`, `pixel_distance(point)`
- `focus(x, y, w, h)`, `focus_reset()`

## Reliable operation order

1. Start from a fresh detection transform or construct a `BoxTransform`.
2. Apply relative focus, if needed.
3. Set the physical anchor, then scale or add directional padding.
4. Fit the square or aspect ratio; restore a non-center anchor if required.
5. Apply translation.
6. Clamp last so reported geometry and extracted pixels agree at boundaries.

## Complete scripts

- `examples/transform_region.py`: model-free manual region composition.
- `examples/extract_detection_crops.py`: class-filtered crops with focus,
  aspect fitting, padding, and a JSON manifest.
- `examples/anchor_zone_filter.py`: annotated contact-anchor zone decisions and
  JSON output.
- `examples/track_zone_events.py`: tracked enter/exit transitions as durable
  JSON Lines records.
- `examples/transform_crop.py`: deterministic synthetic transform smoke test.
