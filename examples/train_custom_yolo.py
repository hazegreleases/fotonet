"""Minimal production training example for a YOLO-format dataset."""

import argparse
from pathlib import Path

from fotonet import Fotonet


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fotonetn")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from <run-dir>/fotonet_last.pt, or provide a checkpoint path.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    checkpoint = None
    if args.resume is not None:
        checkpoint = args.run_dir / "fotonet_last.pt" if args.resume == "auto" else Path(args.resume)

    source = checkpoint or args.model
    print(f"model={source} data={args.data} run_dir={args.run_dir} resume={checkpoint is not None}")
    if args.dry_run:
        Fotonet(source)
        print("Dry run complete; no trainer was created.")
        return 0

    model = Fotonet(source)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        save_dir=str(args.run_dir),
        weights=str(checkpoint) if checkpoint else None,
        resume=checkpoint is not None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
