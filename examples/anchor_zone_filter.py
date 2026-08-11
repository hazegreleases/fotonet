"""Use a detection anchor to decide whether an object is inside a pixel zone."""

from argparse import ArgumentParser
import json
from pathlib import Path

from PIL import Image, ImageDraw

from fotonet import AnchorPoint, BoxTransform, Fotonet


def pil_image(image) -> Image.Image:
    return image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--zone", type=float, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"), help="Zone in source-image pixels.")
    parser.add_argument("--anchor", choices=[item.value for item in AnchorPoint], default="bottom", help="BOTTOM is useful for floor/contact zones.")
    parser.add_argument("--class-name", action="append", default=[], help="Keep this exact class name; repeat for several classes.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("outputs/zone.jpg"))
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    x1, y1, x2, y2 = args.zone
    if x2 <= x1 or y2 <= y1:
        parser.error("--zone requires X2 > X1 and Y2 > Y1")

    model = Fotonet(args.checkpoint)
    result = model.predict(args.image, conf=args.conf, imgsz=args.imgsz, retain_images=True)[0]
    image_w, image_h = result.image_size
    zone = BoxTransform(((x1 + x2) / (2 * image_w), (y1 + y2) / (2 * image_h), (x2 - x1) / image_w, (y2 - y1) / image_h), image_size=result.image_size).clamp()
    canvas = pil_image(result.orig_img).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    zone_tl, zone_br = zone.pixel_corner[1], zone.pixel_corner[3]
    draw.rectangle((zone_tl.x, zone_tl.y, zone_br.x, zone_br.y), outline=(220, 178, 86), width=4)
    records = []
    for box in result.boxes:
        if args.class_name and box.cls not in args.class_name:
            continue
        transform = box.transform.set_anchor(args.anchor)
        point = transform.pixel_position
        accepted = bool(zone.pixel_contains(point=point))
        tl, br = transform.pixel_corner[1], transform.pixel_corner[3]
        color = (51, 190, 112) if accepted else (205, 88, 70)
        draw.rectangle((tl.x, tl.y, br.x, br.y), outline=color, width=3)
        draw.ellipse((point.x - 5, point.y - 5, point.x + 5, point.y + 5), fill=color)
        draw.text((tl.x + 3, max(0, tl.y - 14)), f"{box.cls} {box.conf:.2f}", fill=color)
        records.append({"source_index": box.idx, "class_id": box.cls_id, "class_name": box.cls, "confidence": box.conf, "anchor": transform.anchor.value, "anchor_pixel": [point.x, point.y], "inside_zone": accepted})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    json_output = args.json_output or args.output.with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps({"zone": list(args.zone), "detections": records}, indent=2), encoding="utf-8")
    print(f"accepted {sum(item['inside_zone'] for item in records)} of {len(records)} detections")
    print(f"wrote {args.output.resolve()} and {json_output.resolve()}")


if __name__ == "__main__":
    main()
