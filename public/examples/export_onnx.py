"""Export a trusted FOTO-NET checkpoint to ONNX with its metadata sidecar."""

from argparse import ArgumentParser
from pathlib import Path

from fotonet import Fotonet


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/fotonet.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = Fotonet(args.checkpoint)
    exported = model.export(
        format="onnx",
        path=args.output,
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        verify=True,
    )

    print(f"artifact: {exported['artifact']}")
    print(f"metadata: {exported['metadata']}")


if __name__ == "__main__":
    main()
