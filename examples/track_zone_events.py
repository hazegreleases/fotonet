"""Log tracked objects entering or leaving a pixel zone as JSON Lines."""

from argparse import ArgumentParser
import json
from pathlib import Path

from fotonet import AnchorPoint, BoxTransform, Fotonet


def zone_transform(zone_pixels, image_size) -> BoxTransform:
    x1, y1, x2, y2 = zone_pixels
    image_w, image_h = image_size
    return BoxTransform(((x1 + x2) / (2 * image_w), (y1 + y2) / (2 * image_h), (x2 - x1) / image_w, (y2 - y1) / image_h), image_size=image_size).clamp()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--zone", type=float, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--anchor", choices=[item.value for item in AnchorPoint], default="bottom")
    parser.add_argument("--class-name", action="append", default=[])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tracker-iou", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("outputs/zone-events.jsonl"))
    args = parser.parse_args()
    x1, y1, x2, y2 = args.zone
    if x2 <= x1 or y2 <= y1:
        parser.error("--zone requires X2 > X1 and Y2 > Y1")

    model = Fotonet(args.checkpoint)
    frames = model.track(str(args.video), stream=True, persist=True, conf=args.conf, imgsz=args.imgsz, tracker_iou=args.tracker_iou, max_age=args.max_age, retain_images=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[int, bool] = {}
    event_count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for frame_index, result in enumerate(frames):
            zone = zone_transform(args.zone, result.image_size)
            for box in result.boxes:
                if args.class_name and box.cls not in args.class_name:
                    continue
                if box.track_id is None:
                    continue
                point = box.transform.set_anchor(args.anchor).pixel_position
                inside = bool(zone.pixel_contains(point=point))
                was_inside = previous.get(box.track_id, False)
                previous[box.track_id] = inside
                if inside == was_inside:
                    continue
                record = {"event": "enter" if inside else "exit", "frame": frame_index, "track_id": box.track_id, "class_id": box.cls_id, "class_name": box.cls, "confidence": box.conf, "anchor": args.anchor, "anchor_pixel": [point.x, point.y], "zone": list(args.zone)}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                event_count += 1
    print(f"wrote {event_count} transition events to {args.output.resolve()}")


if __name__ == "__main__":
    main()
