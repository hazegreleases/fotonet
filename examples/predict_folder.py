import argparse
from pathlib import Path

from fotonet import FOTONET


def parse_args():
    parser = argparse.ArgumentParser(description="Run FOTO-NET inference on a folder of images.")
    parser.add_argument("--model", default="fotonet-n", help="Model scale or checkpoint path.")
    parser.add_argument("--source", required=True, help="Folder containing images.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--save-dir", default="dev-tools/runs/example_folder", help="Directory for plotted outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = FOTONET(args.model)
    results = model.predict(args.source, conf=args.conf, imgsz=args.imgsz, batch=args.batch)
    result_list = results if isinstance(results, list) else [results]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(result_list):
        result.save(str(save_dir / f"predict_{index:04d}.jpg"))
    print(f"images={len(result_list)}")
    print(f"saved_dir={save_dir}")


if __name__ == "__main__":
    main()
