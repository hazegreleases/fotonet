"""Crop a hand-specified normalized region with FOTO-NET's transform layer."""

from argparse import ArgumentParser
import json
from pathlib import Path

from PIL import Image

from fotonet import AnchorPoint, BoxTransform


def ratio(value: str) -> tuple[float, float]:
    """Parse W:H or WxH aspect-ratio notation."""
    parts = value.lower().replace("x", ":").split(":")
    if len(parts) != 2:
        raise ValueError("aspect ratio must look like 1:1, 4:3, or 16x9")
    width, height = map(float, parts)
    if width <= 0 or height <= 0:
        raise ValueError("aspect-ratio values must be positive")
    return width, height


def fit_aspect_at_anchor(region: BoxTransform, target: tuple[float, float], mode: int) -> BoxTransform:
    """Fit around the center, then restore the selected anchor in pixels."""
    fixed = region.pixel_position
    region.set_aspect_ratio(target, mode=mode)
    shifted = region.pixel_position
    return region.pixel_move((fixed.x - shifted.x, fixed.y - shifted.y))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--xywh", type=float, nargs=4, required=True,
        metavar=("CX", "CY", "W", "H"),
        help="Normalized center-x, center-y, width, and height.",
    )
    parser.add_argument("--anchor", choices=[item.value for item in AnchorPoint], default="center")
    parser.add_argument("--padding", type=float, default=0.0, help="Pixel padding applied from the active anchor.")
    parser.add_argument("--move", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"), help="Pixel translation after resizing.")
    parser.add_argument("--aspect", type=ratio, help="Optional output shape such as 1:1 or 16:9.")
    parser.add_argument("--fit", choices=("expand", "shrink"), default="expand", help="Expand preserves the whole region; shrink may crop it.")
    parser.add_argument("--output", type=Path, default=Path("outputs/region.png"))
    args = parser.parse_args()

    if args.xywh[2] <= 0 or args.xywh[3] <= 0:
        parser.error("W and H in --xywh must be positive")

    image = Image.open(args.image).convert("RGB")
    region = BoxTransform(tuple(args.xywh), image_size=image.size)
    original_xyxy = region.xyxy

    region.set_anchor(args.anchor).pixel_expand(args.padding)
    if args.aspect:
        fit_aspect_at_anchor(region, args.aspect, mode=1 if args.fit == "expand" else 0)
    region.pixel_move(args.move).clamp()

    crop = region.crop(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(args.output)

    print(json.dumps({
        "input": str(args.image.resolve()),
        "output": str(args.output.resolve()),
        "anchor": region.anchor.value,
        "original_xyxy": list(original_xyxy),
        "active_xyxy": list(region.xyxy),
        "active_pixel_size": list(region.pixel_size),
    }, indent=2))


if __name__ == "__main__":
    main()
