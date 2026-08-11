"""Run one FOTO-NET checkpoint on one image and save annotated output."""

from argparse import ArgumentParser
from pathlib import Path

from fotonet import Fotonet


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.jpg"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = Fotonet(args.checkpoint)
    result = model.predict(args.image, conf=args.conf, imgsz=args.imgsz)[0]

    result.save(args.output)
    print(f"saved {args.output.resolve()}")
    print(result.to_json())


if __name__ == "__main__":
    main()
