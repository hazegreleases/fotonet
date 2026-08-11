"""Friendly command-line training launcher for the public fotonet package."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from typing import Any

from fotonet import Fotonet, __version__


MODELS = (
    "fotonetn", "fotonetn-p2", "fotonets", "fotonets-p2",
    "fotonetm", "fotonetm-p2", "fotonetl", "fotonetl-p2",
    "fotonetx", "fotonetx-p2",
)


def value(text: str) -> Any:
    """Parse Python-like CLI values while preserving ordinary strings."""
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def extra_kwargs(items: list[str]) -> dict[str, Any]:
    signature = inspect.signature(Fotonet.train)
    allowed = {name for name in signature.parameters if name not in {"self", "data"}}
    protected = {"weights", "resume", "pretrained", "save_dir"}
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"Unknown Fotonet.train() option {key!r}. Available: {choices}")
        if key in protected:
            raise ValueError(f"Use the dedicated --{key.replace('_', '-')} option instead of --set")
        parsed[key] = value(raw)
    return parsed


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(
        description="Train or resume a canonical fotonet production V1 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    out.add_argument("--version", action="version", version=f"fotonet {__version__}")
    out.add_argument("--data", type=Path, required=True, help="Dataset YAML with train, val, nc, and ordered names.")
    out.add_argument("--model", choices=MODELS, default="fotonetn")
    out.add_argument("--run-dir", type=Path, help="Output directory; defaults to runs/<model> or the resumed checkpoint directory.")
    out.add_argument("--resume", nargs="?", const="__AUTO__", metavar="CHECKPOINT", help="Resume full state. Bare --resume uses <run-dir>/fotonet_last.pt.")
    out.add_argument("--pretrained", type=Path, help="Start fresh training state from compatible weights.")
    out.add_argument("--recipe", help="Optional packaged recipe name or YAML path.")
    out.add_argument("--epochs", type=int, default=100, help="Total epoch target, not additional epochs on resume.")
    out.add_argument("--imgsz", type=int, default=640)
    out.add_argument("--batch", type=int, default=16)
    out.add_argument("--val-batch", type=int)
    out.add_argument("--workers", type=int)
    out.add_argument("--device", help="Examples: cuda, cuda:0, cpu.")

    optimization = out.add_argument_group("optimization")
    optimization.add_argument("--optimizer", choices=("sgd", "adamw", "musgd"), default="sgd")
    optimization.add_argument("--lr0", type=float, default=0.01)
    optimization.add_argument("--lrf", type=float, default=0.01)
    optimization.add_argument("--momentum", type=float, default=0.937)
    optimization.add_argument("--weight-decay", type=float, default=0.0005)
    optimization.add_argument("--nbs", type=int, default=128, help="Nominal batch size for gradient accumulation and scaling.")
    optimization.add_argument("--warmup-epochs", type=float, default=3.0)
    optimization.add_argument("--frozen-epochs", type=int, default=0)
    optimization.add_argument("--lr-scheduler", choices=("Cosine", "LRDropDown"), default="Cosine")
    optimization.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    data = out.add_argument_group("data and validation")
    data.add_argument("--cache-to-ram", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--annotation-policy", choices=("fix", "error"), default="fix")
    data.add_argument("--allow-missing-labels", action=argparse.BooleanOptionalAction, default=False)
    data.add_argument("--source-recursive", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--val-period", type=int, default=1)
    data.add_argument("--val-conf", type=float, default=0.0)
    data.add_argument("--operating-conf", type=float, default=0.25)
    data.add_argument("--operating-iou", type=float, default=0.50)
    data.add_argument("--coco-max-dets", type=int, default=100)

    output = out.add_argument_group("checkpoints and advanced options")
    output.add_argument("--save-period", type=int, default=-1)
    output.add_argument("--best-metric", choices=("mAP50_95", "mAP50"), default="mAP50_95")
    output.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--slim-best", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--set", dest="settings", action="append", default=[], metavar="KEY=VALUE", help="Forward any validated Fotonet.train() kwarg; repeat as needed.")
    output.add_argument("--dry-run", action="store_true", help="Resolve paths, options, and model/checkpoint identity without training.")
    output.add_argument("-v", "--verbose", action="count", default=1, help="Increase launcher detail; repeat for the complete resolved kwargs.")
    output.add_argument("--quiet", action="store_true", help="Print only errors and the final summary.")
    return out


def resolved_plan(args: argparse.Namespace) -> tuple[Path, Path | None, dict[str, Any]]:
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    if args.resume is not None and args.pretrained is not None:
        raise ValueError("--resume and --pretrained are mutually exclusive")

    explicit_run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    resume_path: Path | None = None
    if args.resume == "__AUTO__":
        if explicit_run_dir is None:
            raise ValueError("Bare --resume requires --run-dir")
        resume_path = explicit_run_dir / "fotonet_last.pt"
    elif args.resume:
        resume_path = Path(args.resume).expanduser().resolve()

    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    pretrained = args.pretrained.expanduser().resolve() if args.pretrained else None
    if pretrained is not None and not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")

    run_dir = explicit_run_dir or (resume_path.parent if resume_path else Path("runs", args.model).resolve())
    kwargs: dict[str, Any] = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "val_batch": args.val_batch,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "nbs": args.nbs,
        "warmup_epochs": args.warmup_epochs,
        "frozen_epochs": args.frozen_epochs,
        "lr_scheduler": args.lr_scheduler,
        "amp": args.amp,
        "cache_to_ram": args.cache_to_ram,
        "pin_memory": args.pin_memory,
        "annotation_policy": args.annotation_policy,
        "allow_missing_labels": args.allow_missing_labels,
        "source_recursive": args.source_recursive,
        "val_period": args.val_period,
        "val_conf": args.val_conf,
        "operating_conf": args.operating_conf,
        "operating_iou": args.operating_iou,
        "coco_max_dets": args.coco_max_dets,
        "save_period": args.save_period,
        "best_metric": args.best_metric,
        "save_last": args.save_last,
        "slim_best": args.slim_best,
        "save_dir": str(run_dir),
    }
    kwargs = {key: item for key, item in kwargs.items() if item is not None}
    kwargs.update(extra_kwargs(args.settings))
    if resume_path is not None:
        kwargs.update(weights=str(resume_path), resume=True)
    elif pretrained is not None:
        kwargs.update(weights=str(pretrained), pretrained=True)
    return run_dir, resume_path or pretrained, kwargs


def main() -> None:
    args = parser().parse_args()
    run_dir, source_checkpoint, kwargs = resolved_plan(args)
    model_source: str | Path = source_checkpoint or args.model

    plan = {
        "fotonet_version": __version__,
        "model": args.model,
        "model_source": str(model_source),
        "data": str(args.data.expanduser().resolve()),
        "run_dir": str(run_dir),
        "mode": "resume" if args.resume is not None else "pretrained" if args.pretrained else "fresh",
        "recipe": args.recipe,
        "train_kwargs": kwargs,
    }
    if not args.quiet:
        print("\n[fotonet] resolved training plan")
        print(json.dumps(plan, indent=2, default=str))

    model = Fotonet(model_source, device=args.device)
    if args.dry_run:
        if not args.quiet:
            print("[fotonet] dry run complete; no dataset or optimizer was constructed.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.quiet:
        print(f"[fotonet] starting {plan['mode']} training in {run_dir}")
    if args.recipe:
        summary = model.train_from_recipe(data=str(args.data), recipe=args.recipe, **kwargs)
    else:
        summary = model.train(data=str(args.data), **kwargs)

    print("\n[fotonet] training finished")
    print(json.dumps(summary, indent=2, default=str) if isinstance(summary, dict) else summary)


if __name__ == "__main__":
    main()
