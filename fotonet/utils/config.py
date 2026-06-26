import os
import yaml

from fotonet.models.scales import normalize_model_config


def _debug_enabled():
    return os.environ.get("FOTONET_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _debug(msg):
    if _debug_enabled():
        print(msg)


def load_model_cfg(cfg_path):
    """
    Load FOTONET model YAML (e.g. fotonetn.yaml).
    Returns dict with: nc, width_multiple, depth_multiple.
    """
    if isinstance(cfg_path, dict):
        return cfg_path
    path = cfg_path
    if not os.path.isabs(path) and not os.path.exists(path):
        _dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(_dir, "config", "models", os.path.basename(cfg_path))
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return normalize_model_config(data or {}, source=path)


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
        return cfg_path

    with open(cfg_path, 'r') as f:
        data = yaml.safe_load(f)

    root = data.get('path', '.')
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    _debug(f"[DEBUG] load_data_cfg: cfg_path={cfg_path}, cfg_dir={cfg_dir}, raw_root={root}")

    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(cfg_dir, root))
    _debug(f"[DEBUG] load_data_cfg: resolved_root={root}")

    for k in ['train', 'val', 'test']:
        if k in data:
            if isinstance(data[k], str):
                data[k] = os.path.normpath(os.path.join(root, data[k]))
            elif isinstance(data[k], list):
                data[k] = [os.path.normpath(os.path.join(root, x)) for x in data[k]]
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
