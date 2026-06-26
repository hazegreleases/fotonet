import argparse
from pathlib import Path

from fotonet import FOTONET


def parse_args():
    parser = argparse.ArgumentParser(description="Run FOTO-NET inference on one image.")
    parser.add_argument("--model", default="fotonet-n", help="Model scale or checkpoint path.")
    parser.add_argument("--source", required=True, help="Image path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size.")
    parser.add_argument("--save", default="dev-tools/runs/example_predict.jpg", help="Output image path. Use an empty value to skip saving.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = FOTONET(args.model)
    results = model.predict(args.source, conf=args.conf, imgsz=args.imgsz)
    print(f"detections={len(results.boxes)}")
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results.save(str(save_path))
        print(f"saved={save_path}")


if __name__ == "__main__":
    main()
