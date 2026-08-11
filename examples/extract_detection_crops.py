"""Detect objects and write reusable, metadata-backed crops."""

from argparse import ArgumentParser
import json
from pathlib import Path
import re

from PIL import Image

from fotonet import AnchorPoint, BoxTransform, Fotonet


def ratio(value: str) -> tuple[float, float]:
    parts = value.lower().replace("x", ":").split(":")
    if len(parts) != 2:
        raise ValueError("aspect ratio must look like 1:1, 4:3, or 16x9")
    width, height = map(float, parts)
    if width <= 0 or height <= 0:
        raise ValueError("aspect-ratio values must be positive")
    return width, height


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "object"


def fit_aspect_at_anchor(region: BoxTransform, target: tuple[float, float]) -> BoxTransform:
    fixed = region.pixel_position
    region.set_aspect_ratio(target, mode=1)
    shifted = region.pixel_position
    return region.pixel_move((fixed.x - shifted.x, fixed.y - shifted.y))


def save_crop(image, path: Path) -> None:
    output = image if isinstance(image, Image.Image) else Image.fromarray(image)
    output.save(path)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/crops"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--class-name", action="append", default=[], help="Keep this exact class name; repeat for several classes.")
    parser.add_argument("--anchor", choices=[item.value for item in AnchorPoint], default="center")
    parser.add_argument("--padding", type=float, default=24.0, help="Context in source-image pixels.")
    parser.add_argument("--aspect", type=ratio, help="Expand each crop to this ratio, for example 1:1 or 4:5.")
    parser.add_argument("--focus", type=float, nargs=4, metavar=("X", "Y", "W", "H"), help="Relative subregion inside each detection; 0 0 1 .55 keeps its upper 55%%.")
    parser.add_argument("--max-crops", type=int, default=0, help="0 keeps every matching detection.")
    args = parser.parse_args()
    if args.focus and (args.focus[0] < 0 or args.focus[1] < 0 or args.focus[2] <= 0 or args.focus[3] <= 0 or args.focus[0] + args.focus[2] > 1 or args.focus[1] + args.focus[3] > 1):
        parser.error("--focus must be a positive x y w h region contained inside 0..1")

    model = Fotonet(args.checkpoint)
    result = model.predict(args.image, conf=args.conf, imgsz=args.imgsz, retain_images=True)[0]
    detections = [box for box in result.boxes if not args.class_name or box.cls in args.class_name]
    detections.sort(key=lambda box: box.conf, reverse=True)
    if args.max_crops > 0:
        detections = detections[:args.max_crops]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, box in enumerate(detections, start=1):
        transform = box.transform
        if args.focus:
            transform.focus(*args.focus)
        transform.set_anchor(args.anchor).pixel_expand(args.padding)
        if args.aspect:
            fit_aspect_at_anchor(transform, args.aspect)
        transform.clamp()
        width, height = transform.pixel_size
        if width < 1 or height < 1:
            continue
        filename = f"{rank:04d}_{safe_name(box.cls)}_{box.conf:.3f}.jpg"
        destination = args.output_dir / filename
        save_crop(transform.crop(result.orig_img), destination)
        manifest.append({"file": filename, "source_index": box.idx, "class_id": box.cls_id, "class_name": box.cls, "confidence": box.conf, "detection_xywh": list(box.xywh), "crop_xyxy": list(transform.xyxy), "crop_pixel_size": [width, height], "anchor": transform.anchor.value, "focus": args.focus})
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest)} crops and {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
