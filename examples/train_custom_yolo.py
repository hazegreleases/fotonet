import argparse
from pathlib import Path

from fotonet import FOTONET


def parse_args():
    parser = argparse.ArgumentParser(description="Train FOTO-NET on a YOLO-format dataset.")
    parser.add_argument("--model", default="fotonetn", help="Model scale or checkpoint path.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--save-dir", default="dev-tools/runs/train", help="Training output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")
    model = FOTONET(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
