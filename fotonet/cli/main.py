"""FOTO-NET command line interface.

Examples:
  fotonet train model=fotonetn data=datasets/coco/coco.yaml epochs=250 batch=16 imgsz=640
  fotonet predict model=fotonet-n source=image.jpg conf=0.25 save=true
  fotonet val model=fotonet-n data=datasets/coco/coco.yaml imgsz=640
  fotonet export model=fotonet-n format=onnx path=dev-tools/runs/fotonet.onnx half=true

YOLO-style mode form is also accepted:
  fotonet mode=train model=fotonetn data=coco.yaml epochs=100
"""

import ast
import os
import sys


TASK_ALIASES = {
    "detect": "predict",
    "infer": "predict",
    "predict": "predict",
    "track": "track",
    "train": "train",
    "val": "val",
    "validate": "val",
    "export": "export",
}


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

    task = args.pop("mode", args.pop("task", task or "predict"))
    task = TASK_ALIASES.get(str(task).lower())
    if task is None:
        valid = ", ".join(sorted(set(TASK_ALIASES.values())))
        raise SystemExit(f"Invalid task. Use one of: {valid}.")
    return task, args


def _pop_bool(args, key, default=False):
    value = args.pop(key, default)
    return bool(value)


def _print_help():
    print(__doc__.strip())


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    from fotonet import FOTONET

    task, args = parse_args(argv)
    model_name = args.pop("model", "fotonetn")
    imgsz = int(args.pop("imgsz", 640))

    model = FOTONET(model_name)

    if task == "train":
        data = args.pop("data", None)
        if not data:
            raise SystemExit("Training requires data=path/to/data.yaml.")
        epochs = int(args.pop("epochs", 100))
        save_path = args.pop("save", "fotonet_trained.pt")
        print(f"FOTO-NET train | model={model_name} data={data} epochs={epochs} imgsz={imgsz}")
        model.train(data=data, epochs=epochs, imgsz=imgsz, **args)
        if save_path:
            model.save(str(save_path), inference_only=True, half=True)
            print(f"saved={save_path}")
        return

    if task in {"predict", "track"}:
        source = args.pop("source", None)
        if not source:
            raise SystemExit(f"{task} requires source=image.jpg|folder.")
        conf = float(args.pop("conf", 0.25))
        save = _pop_bool(args, "save", True)
        show = _pop_bool(args, "show", False)
        use_nms = _pop_bool(args, "use_nms", False)
        results = model.predict(source, imgsz=imgsz, conf=conf, use_nms=use_nms, **args)
        result_list = results if isinstance(results, list) else [results]
        print(
            f"FOTO-NET {task} | model={model_name} source={source} "
            f"images={len(result_list)} nms_free={not use_nms}"
        )
        if save:
            save_dir = "runs"
            os.makedirs(save_dir, exist_ok=True)
            for i, result in enumerate(result_list):
                result.save(os.path.join(save_dir, f"predict_{i}.jpg"))
            print(f"saved_dir={save_dir}")
        if show and not isinstance(results, list):
            results.show()
        return

    if task == "val":
        data = args.pop("data", None)
        if not data:
            raise SystemExit("Validation requires data=path/to/data.yaml.")
        metrics = model.val(data=data, imgsz=imgsz, **args)
        print("FOTO-NET val")
        for key, value in metrics.items():
            if key.endswith("per_class"):
                continue
            print(f"{key}={value}")
        return

    if task == "export":
        fmt = str(args.pop("format", "onnx")).lower()
        path = args.pop("path", f"fotonet_export.{fmt}")
        out = model.export(path=path, format=fmt, imgsz=imgsz, **args)
        print(f"exported={out}")


if __name__ == "__main__":
    main()
