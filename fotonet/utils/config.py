import os
from numbers import Integral
from urllib.parse import urlparse
import yaml

from fotonet.models.v1.registry import normalize_model_config


def normalize_class_schema(nc=None, names=None, *, context="class schema"):
    """Return a strict, canonical ``(nc, names)`` class schema.

    A detector class index is a semantic contract, not just an output width.
    Keeping one canonical ``{class_id: name}`` representation lets data configs,
    checkpoints, and public APIs compare class order safely instead of merely
    comparing the number of classes.

    ``names`` may be a YOLO-style sequence or a mapping (including YAML's
    stringified integer keys).  When names are present and ``nc`` is omitted,
    the class count is inferred from the names.  If neither is supplied the
    caller can still choose its own model default, so ``(None, None)`` is
    returned.
    """
    normalized_nc = None
    if nc is not None:
        if isinstance(nc, bool) or not isinstance(nc, Integral):
            raise TypeError(f"{context}.nc must be a positive integer, got {nc!r}")
        normalized_nc = int(nc)
        if normalized_nc <= 0:
            raise ValueError(f"{context}.nc must be a positive integer, got {nc!r}")

    normalized_names = None
    if names is not None:
        if isinstance(names, dict):
            raw_items = names.items()
        elif isinstance(names, (list, tuple)):
            raw_items = enumerate(names)
        else:
            raise TypeError(
                f"{context}.names must be a mapping or a list/tuple of class names, "
                f"got {type(names).__name__}"
            )

        normalized_names = {}
        for raw_index, raw_name in raw_items:
            if isinstance(raw_index, bool):
                raise ValueError(f"{context}.names keys must be integer class IDs, got {raw_index!r}")
            if isinstance(raw_index, Integral):
                index = int(raw_index)
            elif isinstance(raw_index, str):
                try:
                    index = int(raw_index)
                except ValueError as exc:
                    raise ValueError(
                        f"{context}.names keys must be integer class IDs, got {raw_index!r}"
                    ) from exc
            else:
                raise ValueError(f"{context}.names keys must be integer class IDs, got {raw_index!r}")
            if not isinstance(raw_name, str):
                raise TypeError(
                    f"{context}.names[{raw_index!r}] must be a non-empty string, "
                    f"got {type(raw_name).__name__}"
                )
            name = raw_name.strip()
            if not name:
                raise ValueError(f"{context}.names[{raw_index!r}] must be a non-empty string")
            if index in normalized_names:
                raise ValueError(f"{context}.names contains duplicate class ID {index}")
            normalized_names[index] = name

        inferred_nc = len(normalized_names)
        if normalized_nc is None:
            normalized_nc = inferred_nc
            if normalized_nc <= 0:
                raise ValueError(f"{context}.names must define at least one class")
        elif inferred_nc != normalized_nc:
            raise ValueError(
                f"{context}.nc={normalized_nc} does not match {inferred_nc} class name(s)"
            )
        expected_ids = set(range(normalized_nc))
        actual_ids = set(normalized_names)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ValueError(
                f"{context}.names must cover exactly class IDs 0..{normalized_nc - 1}; "
                f"missing={missing}, extra={extra}"
            )
        normalized_names = {index: normalized_names[index] for index in range(normalized_nc)}

    return normalized_nc, normalized_names


def default_class_names(nc):
    """Return explicit placeholder names for a known class count."""
    normalized_nc, _ = normalize_class_schema(nc, context="model")
    return {index: f"class_{index}" for index in range(normalized_nc)}


def class_names_are_default(names, nc):
    """Whether names are only the generated ``class_<id>`` placeholders."""
    normalized_nc, normalized_names = normalize_class_schema(nc, names, context="model")
    return normalized_names == default_class_names(normalized_nc)


def _debug_enabled():
    return os.environ.get("FOTONET_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _debug(msg):
    if _debug_enabled():
        print(msg)


def _is_remote_path(value):
    """Do not turn an HTTP URI into a bogus local path."""
    if not isinstance(value, (str, os.PathLike)):
        return False
    return urlparse(os.fspath(value)).scheme in {"http", "https"}


def _resolve_data_path(value, root):
    """Resolve one dataset source while preserving direct COCO source mappings."""
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if isinstance(value, str):
        if _is_remote_path(value) or os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(root, value))
    if isinstance(value, (list, tuple)):
        return [_resolve_data_path(item, root) for item in value]
    if isinstance(value, dict):
        resolved = dict(value)
        for key in ("images", "image_root", "image_dir", "annotations", "annotation", "json"):
            if key in resolved:
                resolved[key] = _resolve_data_path(resolved[key], root)
        return resolved
    raise TypeError(f"Dataset source must be a path, list, or mapping; got {type(value).__name__}")


def _resolve_data_cfg(data, root):
    resolved = dict(data or {})
    for key in ("train", "val", "test", "train_images", "val_images", "test_images", "coco_images"):
        if key in resolved:
            resolved[key] = _resolve_data_path(resolved[key], root)
    nc, names = normalize_class_schema(
        resolved.get("nc"),
        resolved.get("names"),
        context="data config",
    )
    if nc is not None:
        resolved["nc"] = nc
    if names is not None:
        resolved["names"] = names
    return resolved


def load_model_cfg(cfg_path):
    """Load one canonical fotonet model YAML."""
    if isinstance(cfg_path, dict):
        return cfg_path
    path = cfg_path
    if not os.path.isabs(path) and not os.path.exists(path):
        _dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(_dir, "config", "models", os.path.basename(cfg_path))
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return normalize_model_config(
        data or {}, model_id=os.path.splitext(os.path.basename(path))[0], source=path
    )


def load_data_cfg(cfg_path):
    """
    Loads and parses a YOLO-style data.yaml file.
    Example:
    path: ../datasets/coco128
    train: images/train2017
    val: images/val2017
    nc: 80
    names: [ 'person', 'bicycle', ... ]
    """
    if isinstance(cfg_path, dict):
        data = dict(cfg_path)
        root = data.get("path", ".")
        if isinstance(root, os.PathLike):
            root = os.fspath(root)
        if not isinstance(root, str):
            raise TypeError("data.path must be a filesystem path string")
        if not os.path.isabs(root):
            root = os.path.abspath(root)
        return _resolve_data_cfg(data, root)

    with open(cfg_path, 'r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Data YAML must contain a mapping at its top level")

    root = data.get('path', '.')
    if isinstance(root, os.PathLike):
        root = os.fspath(root)
    if not isinstance(root, str):
        raise TypeError("data.path must be a filesystem path string")
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    _debug(f"[DEBUG] load_data_cfg: cfg_path={cfg_path}, cfg_dir={cfg_dir}, raw_root={root}")

    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(cfg_dir, root))
    _debug(f"[DEBUG] load_data_cfg: resolved_root={root}")

    data = _resolve_data_cfg(data, root)
    for k in ['train', 'val', 'test']:
        if k in data:
            _debug(f"[DEBUG] load_data_cfg: {k}_path={data[k]}")

    return data

def get_label_path(img_path):
    """
    Converts image path to label path using several common patterns.
    1. Standard YOLO: /images/train/img1.png -> /labels/train/img1.txt
    2. Nested COCO: /datasets/coco/train2017/img1.png -> /datasets/coco/coco/labels/train2017/img1.txt
    3. Flat sibling: /train/img1.png -> /labels/train/img1.txt
    """
    img_path = os.path.normpath(img_path)
    abs_img = os.path.abspath(img_path)
    
    # 1. Standard pattern: replace 'images' with 'labels'
    if 'images' in abs_img:
        parts = abs_img.split(os.sep)
        label_parts = [p if p != 'images' else 'labels' for p in parts]
        lp = os.sep.join(label_parts)
        lp = os.path.splitext(lp)[0] + '.txt'
        if os.path.exists(lp):
            return lp

    # 2. Nested COCO pattern: find the dataset root and look for 'coco/labels' or 'labels'
    # Specifically for the user's case: .../datasets/coco/train2017 -> .../datasets/coco/coco/labels/train2017
    parts = abs_img.split(os.sep)
    for i in range(len(parts)-1, 0, -1):
        if parts[i] in ['train2017', 'val2017', 'test2017', 'train128']:
            subset = parts[i]
            # Try prepending 'coco/labels' or 'labels' at the directory level above subset
            root_dir = os.sep.join(parts[:i])
            for label_dir in ['labels', os.path.join('coco', 'labels'), 'Annotations']:
                lp = os.path.join(root_dir, label_dir, subset, os.path.basename(abs_img))
                lp = os.path.splitext(lp)[0] + '.txt'
                if os.path.exists(lp):
                    return lp

    # 3. Fallback: try same directory but .txt extension
    lp = os.path.splitext(abs_img)[0] + '.txt'
    return lp
