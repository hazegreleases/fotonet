"""Validate a trusted fotonet checkpoint against an explicit dataset YAML."""

from argparse import ArgumentParser
from pathlib import Path

from fotonet import Fotonet


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = Fotonet(args.checkpoint)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.0,
        max_det=100,
    )

    for name, value in metrics.items():
        if not name.endswith("per_class"):
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
