"""Run folder inference and write one JSON record per source image."""

from argparse import ArgumentParser
from pathlib import Path

from fotonet import Fotonet


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/detections.jsonl"))
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    model = Fotonet(args.checkpoint)
    results = model.predict(args.images, batch=args.batch, conf=0.25, imgsz=640)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(result.to_json() + "\n")

    print(f"wrote {len(results)} image records to {args.output.resolve()}")


if __name__ == "__main__":
    main()
