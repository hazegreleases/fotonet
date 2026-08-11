import argparse
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run FOTO-NET inference on one image.")
    parser.add_argument("--model", required=True, help="Self-identifying checkpoint path.")
    parser.add_argument("--source", help="Optional image path. If omitted, a file picker opens.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size.")
    parser.add_argument(
        "--save",
        default="runs/examples",
        help="Output directory or exact output image path. Use an empty value to skip saving.",
    )
    return parser.parse_args(argv)


def choose_image():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        selected = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def resolve_save_path(save, image_path):
    if not save:
        return None
    image_path = Path(image_path)
    save_path = Path(save)
    if save_path.suffix:
        return save_path
    return save_path / f"{image_path.stem}_predict.jpg"


def save_prediction(results, save_path):
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results.save(str(save_path))
    print(f"saved={save_path}")


def show_preview(results, image_path):
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required for interactive preview. Install opencv-python.") from exc

    window_name = f"FOTO-NET: {Path(image_path).name} | q quit | r another"
    plot_img = results.plot()
    cv2.imshow(window_name, cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in {ord("q"), 27}:
            cv2.destroyAllWindows()
            return "quit"
        if key == ord("r"):
            cv2.destroyAllWindows()
            return "repeat"


def run_picker_loop(initial_source, pick_image, predict, save, show):
    current = Path(initial_source) if initial_source else None
    while True:
        if current is None:
            current = pick_image()
            if current is None:
                print("No image selected.")
                return "cancelled"
            current = Path(current)

        results = predict(current)
        print(f"detections={len(results.boxes)}")
        save(results, current)
        action = show(results, current)
        if action in {"repeat", "r"}:
            current = None
            continue
        return "quit"


def main(argv=None):
    args = parse_args(argv)

    from fotonet import Fotonet

    model = Fotonet(args.model)

    def predict(path):
        return model.predict(str(path), conf=args.conf, imgsz=args.imgsz)[0]

    def save(results, path):
        save_prediction(results, resolve_save_path(args.save, path))

    run_picker_loop(
        initial_source=args.source,
        pick_image=choose_image,
        predict=predict,
        save=save,
        show=show_preview,
    )


if __name__ == "__main__":
    main()
