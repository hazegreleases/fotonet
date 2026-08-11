"""fotonet command line interface.

Examples:
  fotonet train model=fotonetn data=datasets/coco/coco.yaml epochs=250 batch=16 imgsz=640
  fotonet predict model=weights/fotonetn.pt source=image.jpg conf=0.25 save=true
  fotonet val model=weights/fotonetn.pt data=datasets/coco/coco.yaml imgsz=640
  fotonet export model=weights/fotonetn.pt format=onnx path=fotonet.onnx
"""

import ast
import os
import sys
from importlib.metadata import PackageNotFoundError, version


TASKS = {"train", "predict", "track", "val", "export"}


def _coerce(value):
    raw = str(value).strip()
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return value


def parse_args(argv):
    args = {}
    task = None
    remaining = list(argv)

    if remaining and "=" not in remaining[0]:
        task = remaining.pop(0).lower()

    for token in remaining:
        if "=" not in token:
            raise SystemExit(f"Invalid argument '{token}'. Use key=value.")
        key, value = token.split("=", 1)
        args[key.replace("-", "_")] = _coerce(value)

    task = str(task or "predict").lower()
    if task not in TASKS:
        raise SystemExit(f"Invalid task. Use one of: {', '.join(sorted(TASKS))}.")
    return task, args


def _pop_bool(args, key, default=False):
    value = args.pop(key, default)
    return bool(value)


def _parse_imgsz(value):
    """Accept the public scalar or (height, width) image-size form."""
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit("imgsz must be an integer or an (height,width) pair, e.g. imgsz=(640,960).") from exc


def _print_help():
    print(__doc__.strip())


def _version_text():
    try:
        from fotonet import __version__ as package_version
    except ImportError:
        try:
            package_version = version("fotonet")
        except PackageNotFoundError:
            package_version = "unknown"
    return f"fotonet {package_version}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return
    if argv[0] in {"-V", "--version", "version"}:
        print(_version_text())
        return

    from fotonet import Fotonet

    task, args = parse_args(argv)
    model_name = args.pop("model", None)
    if not model_name:
        raise SystemExit("Every task requires model=fotonetn or model=path/to/checkpoint.pt.")
    imgsz = _parse_imgsz(args.pop("imgsz", 640))

    model = Fotonet(model_name)

    if task == "train":
        data = args.pop("data", None)
        if not data:
            raise SystemExit("Training requires data=path/to/data.yaml.")
        recipe = args.pop("recipe", None)
        epochs = int(args.pop("epochs", 100)) if recipe is None else None
        save_path = args.pop("save", "fotonet_trained.pt")
        if recipe is None:
            print(f"fotonet train | model={model_name} data={data} epochs={epochs} imgsz={imgsz}")
            result = model.train(data=data, epochs=epochs, imgsz=imgsz, **args)
        else:
            print(f"fotonet train | model={model_name} data={data} recipe={recipe}")
            result = model.train_from_recipe(data=data, recipe=recipe, imgsz=imgsz, **args)
        if save_path:
            best_path = result.get("best_checkpoint") if isinstance(result, dict) else None
            if best_path:
                print(f"best_checkpoint={best_path}")
            else:
                model.save(str(save_path), inference_only=True, half=True)
                print(f"saved={save_path}")
        return

    if task in {"predict", "track"}:
        source = args.pop("source", None)
        if source is None or source == "":
            raise SystemExit(f"{task} requires source=image.jpg|folder|video|webcam-index.")
        conf = float(args.pop("conf", 0.25))
        save = _pop_bool(args, "save", True)
        show = _pop_bool(args, "show", False)
        runner = model.track if task == "track" else model.predict
        results = runner(source, imgsz=imgsz, conf=conf, **args)
        # Do not materialize video/webcam generators: a long stream should not
        # accumulate every Result in RAM just because ``save`` is enabled.
        result_iter = iter(results) if not isinstance(results, list) else iter(results)
        save_dir = "runs" if save else None
        if save:
            os.makedirs(save_dir, exist_ok=True)
        count = 0
        for result in result_iter:
            if save:
                result.save(os.path.join(save_dir, f"{task}_{count}.jpg"))
            if show:
                result.show()
            count += 1
        print(
            f"fotonet {task} | model={model_name} source={source} images={count}"
        )
        if save:
            print(f"saved_dir={save_dir}")
        return

    if task == "val":
        data = args.pop("data", None)
        if not data:
            raise SystemExit("Validation requires data=path/to/data.yaml.")
        metrics = model.val(data=data, imgsz=imgsz, **args)
        print("fotonet val")
        for key, value in metrics.items():
            if key.endswith("per_class"):
                continue
            print(f"{key}={value}")
        return

    if task == "export":
        fmt = str(args.pop("format", "onnx")).lower()
        path = args.pop("path", f"fotonet_export.{fmt}")
        out = model.export(path=path, format=fmt, imgsz=imgsz, **args)
        print(f"exported={out['artifact']}")
        print(f"metadata={out['metadata']}")


if __name__ == "__main__":
    main()
