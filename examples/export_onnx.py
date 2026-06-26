import argparse

from fotonet import FOTONET


def parse_args():
    parser = argparse.ArgumentParser(description="Export FOTO-NET to ONNX.")
    parser.add_argument("--model", default="fotonet-n", help="Model scale or checkpoint path.")
    parser.add_argument("--path", default="dev-tools/runs/fotonet.onnx", help="Output ONNX path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size.")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size.")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic batch axis.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = FOTONET(args.model)
    artifact = model.export(
        format="onnx",
        path=args.path,
        imgsz=args.imgsz,
        batch=args.batch,
        dynamic=args.dynamic,
    )
    print(f"exported={artifact}")


if __name__ == "__main__":
    main()
