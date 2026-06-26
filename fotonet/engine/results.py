import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from PIL import Image

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


@dataclass(frozen=True)
class Vector2:
    x: float
    y: float

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, idx):
        return (self.x, self.y)[idx]


class AnchorPoint(Enum):
    CENTER = "center"
    DEFAULT = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


def _normalize_anchor(anchor):
    if isinstance(anchor, AnchorPoint):
        return anchor
    name = str(anchor).strip()
    if name.lower().startswith("anchorpoint."):
        name = name.split(".", 1)[1]
    if name.upper() in AnchorPoint.__members__:
        return AnchorPoint[name.upper()]
    return AnchorPoint(name.lower())


@dataclass(frozen=True)
class FocusRegion:
    x: float
    y: float
    w: float
    h: float

    def blend(self, other, alpha=0.5):
        return FocusRegion(
            self.x * (1.0 - alpha) + other.x * alpha,
            self.y * (1.0 - alpha) + other.y * alpha,
            self.w * (1.0 - alpha) + other.w * alpha,
            self.h * (1.0 - alpha) + other.h * alpha,
        )

    def __truediv__(self, other):
        return self.blend(other, 0.5)


class _PointMap:
    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, idx):
        return self._fn(idx)


class _FocusManager:
    def __init__(self, transform):
        self._transform = transform

    def __call__(self, x, y, w, h):
        region = FocusRegion(float(x), float(y), float(w), float(h))
        self._transform._apply_focus(region)
        return region

    def Reset(self):
        self._transform.focusReset()
        return self._transform


def _anchor_position_xyxy(xyxy, anchor):
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    if anchor == AnchorPoint.TOP_LEFT:
        return x1, y1
    if anchor == AnchorPoint.TOP_RIGHT:
        return x2, y1
    if anchor == AnchorPoint.BOTTOM_LEFT:
        return x1, y2
    if anchor == AnchorPoint.BOTTOM_RIGHT:
        return x2, y2
    if anchor == AnchorPoint.TOP:
        return cx, y1
    if anchor == AnchorPoint.BOTTOM:
        return cx, y2
    if anchor == AnchorPoint.LEFT:
        return x1, cy
    if anchor == AnchorPoint.RIGHT:
        return x2, cy
    return cx, cy


class BoxTransform:
    """Pure spatial utility layer over one detection box."""

    def __init__(self, box_xywh, image_size):
        self._source = np.asarray(box_xywh, dtype=np.float32).copy()
        self._box = self._source.copy()
        self._image_w, self._image_h = image_size
        self._anchor = AnchorPoint.CENTER
        self._focus_manager = _FocusManager(self)

    @property
    def Anchor(self):
        return self._anchor

    @Anchor.setter
    def Anchor(self, anchor):
        self._anchor = _normalize_anchor(anchor)

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, anchor):
        self.Anchor = anchor

    @property
    def focus(self):
        return self._focus_manager

    @focus.setter
    def focus(self, region):
        if isinstance(region, _FocusManager):
            self._focus_manager = region
            return
        if region is None:
            self.focusReset()
            return
        if isinstance(region, FocusRegion):
            self._apply_focus(region)
            return
        if isinstance(region, (tuple, list)) and len(region) == 4:
            self._apply_focus(FocusRegion(*map(float, region)))
            return
        raise TypeError("focus must be a FocusRegion, (x, y, w, h), None, or focus manager.")

    def _xyxy(self):
        x, y, w, h = self._box
        return np.array([x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5], dtype=np.float32)

    def _set_xyxy(self, xyxy):
        x1, y1, x2, y2 = np.asarray(xyxy, dtype=np.float32)
        self._box = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(x2 - x1, 0.0), max(y2 - y1, 0.0)], dtype=np.float32)

    def _pixel_vec(self, v):
        return Vector2(v.x * self._image_w, v.y * self._image_h)

    @property
    def xywh(self):
        return tuple(float(v) for v in self._box)

    @property
    def xyxy(self):
        return tuple(float(v) for v in self._xyxy())

    @property
    def position(self):
        x, y = _anchor_position_xyxy(self._xyxy(), self.Anchor)
        return Vector2(float(x), float(y))

    @property
    def pixelPosition(self):
        return self._pixel_vec(self.position)

    @property
    def size(self):
        return Vector2(float(self._box[2]), float(self._box[3]))

    @property
    def pixelSize(self):
        return Vector2(float(self._box[2] * self._image_w), float(self._box[3] * self._image_h))

    @property
    def corner(self):
        return _PointMap(self._corner)

    @property
    def pixelCorner(self):
        return _PointMap(lambda idx: self._pixel_vec(self._corner(idx)))

    @property
    def side(self):
        return _PointMap(self._side)

    @property
    def pixelSide(self):
        return _PointMap(lambda idx: self._pixel_vec(self._side(idx)))

    def _corner(self, idx):
        x1, y1, x2, y2 = self._xyxy()
        pts = {
            1: Vector2(float(x1), float(y1)),
            2: Vector2(float(x2), float(y1)),
            3: Vector2(float(x2), float(y2)),
            4: Vector2(float(x1), float(y2)),
        }
        return pts[int(idx)]

    def _side(self, idx):
        x1, y1, x2, y2 = self._xyxy()
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        pts = {
            1: Vector2(float(cx), float(y1)),
            2: Vector2(float(x2), float(cy)),
            3: Vector2(float(cx), float(y2)),
            4: Vector2(float(x1), float(cy)),
        }
        return pts[int(idx)]

    def setAnchor(self, anchor):
        self.Anchor = anchor
        return self

    def move(self, delta):
        dx, dy = delta if not isinstance(delta, Vector2) else (delta.x, delta.y)
        self._box[0] += float(dx)
        self._box[1] += float(dy)
        return self

    def pixelMove(self, delta):
        dx, dy = delta if not isinstance(delta, Vector2) else (delta.x, delta.y)
        return self.move(Vector2(float(dx) / self._image_w, float(dy) / self._image_h))

    def _scale_about_anchor(self, sx, sy):
        x1, y1, x2, y2 = self._xyxy()
        ax, ay = _anchor_position_xyxy((x1, y1, x2, y2), self.Anchor)
        x1 = ax + (x1 - ax) * sx
        x2 = ax + (x2 - ax) * sx
        y1 = ay + (y1 - ay) * sy
        y2 = ay + (y2 - ay) * sy
        self._set_xyxy((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    def scale(self, factor):
        sx, sy = factor if not isinstance(factor, Vector2) else (factor.x, factor.y)
        self._scale_about_anchor(float(sx), float(sy))
        return self

    def expand(self, padding):
        p = float(padding)
        x1, y1, x2, y2 = self._xyxy()
        if self.Anchor in (AnchorPoint.CENTER, AnchorPoint.TOP, AnchorPoint.BOTTOM):
            x1 -= p
            x2 += p
        elif self.Anchor in (AnchorPoint.LEFT, AnchorPoint.TOP_LEFT, AnchorPoint.BOTTOM_LEFT):
            x2 += p
        elif self.Anchor in (AnchorPoint.RIGHT, AnchorPoint.TOP_RIGHT, AnchorPoint.BOTTOM_RIGHT):
            x1 -= p
        if self.Anchor in (AnchorPoint.CENTER, AnchorPoint.LEFT, AnchorPoint.RIGHT):
            y1 -= p
            y2 += p
        elif self.Anchor in (AnchorPoint.TOP, AnchorPoint.TOP_LEFT, AnchorPoint.TOP_RIGHT):
            y2 += p
        elif self.Anchor in (AnchorPoint.BOTTOM, AnchorPoint.BOTTOM_LEFT, AnchorPoint.BOTTOM_RIGHT):
            y1 -= p
        self._set_xyxy((x1, y1, x2, y2))
        return self

    def pixelExpand(self, padding):
        p = float(padding)
        px = p / max(float(self._image_w), 1e-9)
        py = p / max(float(self._image_h), 1e-9)
        x1, y1, x2, y2 = self._xyxy()
        if self.Anchor in (AnchorPoint.CENTER, AnchorPoint.TOP, AnchorPoint.BOTTOM):
            x1 -= px
            x2 += px
        elif self.Anchor in (AnchorPoint.LEFT, AnchorPoint.TOP_LEFT, AnchorPoint.BOTTOM_LEFT):
            x2 += px
        elif self.Anchor in (AnchorPoint.RIGHT, AnchorPoint.TOP_RIGHT, AnchorPoint.BOTTOM_RIGHT):
            x1 -= px
        if self.Anchor in (AnchorPoint.CENTER, AnchorPoint.LEFT, AnchorPoint.RIGHT):
            y1 -= py
            y2 += py
        elif self.Anchor in (AnchorPoint.TOP, AnchorPoint.TOP_LEFT, AnchorPoint.TOP_RIGHT):
            y2 += py
        elif self.Anchor in (AnchorPoint.BOTTOM, AnchorPoint.BOTTOM_LEFT, AnchorPoint.BOTTOM_RIGHT):
            y1 -= py
        self._set_xyxy((x1, y1, x2, y2))
        return self

    def clamp(self):
        x1, y1, x2, y2 = self._xyxy()
        self._set_xyxy((np.clip(x1, 0, 1), np.clip(y1, 0, 1), np.clip(x2, 0, 1), np.clip(y2, 0, 1)))
        return self

    def setBox(self, mode=1):
        x, y, w, h = self._box
        side = min(w, h) if int(mode) == 0 else max(w, h)
        self._box = np.array([x, y, side, side], dtype=np.float32)
        return self

    def setAspectRatio(self, ratio, mode=1):
        rw, rh = ratio if not isinstance(ratio, Vector2) else (ratio.x, ratio.y)
        target = max(float(rw), 1e-9) / max(float(rh), 1e-9)
        x, y, w, h = self._box
        if int(mode) == 0:
            if w / max(h, 1e-9) > target:
                w = h * target
            else:
                h = w / target
        else:
            if w / max(h, 1e-9) > target:
                h = w / target
            else:
                w = h * target
        self._box = np.array([x, y, w, h], dtype=np.float32)
        return self

    def crop(self, image):
        x1, y1, x2, y2 = self._xyxy()
        px1 = int(np.clip(round(x1 * self._image_w), 0, self._image_w))
        py1 = int(np.clip(round(y1 * self._image_h), 0, self._image_h))
        px2 = int(np.clip(round(x2 * self._image_w), 0, self._image_w))
        py2 = int(np.clip(round(y2 * self._image_h), 0, self._image_h))
        if isinstance(image, Image.Image):
            return image.crop((px1, py1, px2, py2))
        return image[py1:py2, px1:px2].copy()

    def contains(self, point=None, x=None, y=None, threshold=0.0):
        x1, y1, x2, y2 = self._xyxy()
        if point is not None:
            px, py = point if not isinstance(point, Vector2) else (point.x, point.y)
            return x1 <= px <= x2 and y1 <= py <= y2
        zx1, zx2 = x if not isinstance(x, Vector2) else (x.x, x.y)
        zy1, zy2 = y if not isinstance(y, Vector2) else (y.x, y.y)
        ix = max(0.0, min(x2, zx2) - max(x1, zx1))
        iy = max(0.0, min(y2, zy2) - max(y1, zy1))
        area = max((x2 - x1) * (y2 - y1), 1e-9)
        return (ix * iy) / area >= float(threshold)

    def pixelContains(self, point=None, x=None, y=None, threshold=0.0):
        if point is not None:
            px, py = point if not isinstance(point, Vector2) else (point.x, point.y)
            return self.contains(Vector2(px / self._image_w, py / self._image_h))
        zx1, zx2 = x if not isinstance(x, Vector2) else (x.x, x.y)
        zy1, zy2 = y if not isinstance(y, Vector2) else (y.x, y.y)
        return self.contains(x=Vector2(zx1 / self._image_w, zx2 / self._image_w),
                             y=Vector2(zy1 / self._image_h, zy2 / self._image_h),
                             threshold=threshold)

    def overlaps(self, other):
        return self.iou(other) > 0.0

    def iou(self, other):
        other_xyxy = other.transform.xyxy if hasattr(other, "transform") else other.xyxy
        x1, y1, x2, y2 = self._xyxy()
        ox1, oy1, ox2, oy2 = other_xyxy
        inter = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
        area1 = max((x2 - x1) * (y2 - y1), 0.0)
        area2 = max((ox2 - ox1) * (oy2 - oy1), 0.0)
        return float(inter / max(area1 + area2 - inter, 1e-9))

    def distance(self, point):
        px, py = point if not isinstance(point, Vector2) else (point.x, point.y)
        pos = self.position
        return float(((pos.x - px) ** 2 + (pos.y - py) ** 2) ** 0.5)

    def pixelDistance(self, point):
        px, py = point if not isinstance(point, Vector2) else (point.x, point.y)
        pos = self.pixelPosition
        return float(((pos.x - px) ** 2 + (pos.y - py) ** 2) ** 0.5)

    def _apply_focus(self, region):
        x1, y1, x2, y2 = self._xyxy()
        bw = x2 - x1
        bh = y2 - y1
        fx = x1 + region.x * bw
        fy = y1 + region.y * bh
        fw = region.w * bw
        fh = region.h * bh
        self._box = np.array([fx, fy, fw, fh], dtype=np.float32)

    def focusReset(self):
        self._box = self._source.copy()
        return self


class DetectionBox:
    def __init__(self, parent, idx):
        self._parent = parent
        self.idx = int(idx)
        self.conf = float(parent.scores[idx])
        self.cls_id = int(parent.classes[idx])
        self.cls = parent.names.get(self.cls_id, str(self.cls_id))
        self.xywh = tuple(float(v) for v in parent.tensor[idx].tolist())
        self.transform = BoxTransform(self.xywh, parent.image_size)

    @property
    def xyxy(self):
        return self.transform.xyxy


class DetectionBoxes:
    def __init__(self, tensor, scores, classes, names=None, image_size=(640, 640)):
        self.tensor = tensor.detach().cpu() if torch.is_tensor(tensor) else torch.as_tensor(tensor)
        self.scores = scores.detach().cpu() if torch.is_tensor(scores) else torch.as_tensor(scores)
        self.classes = classes.detach().cpu() if torch.is_tensor(classes) else torch.as_tensor(classes)
        self.names = names or {}
        self.image_size = image_size

    def __len__(self):
        return int(self.tensor.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield DetectionBox(self, i)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [DetectionBox(self, i) for i in range(*idx.indices(len(self)))]
        return DetectionBox(self, idx)

    def numpy(self):
        return self.tensor.numpy()

    def to_tensor(self):
        return self.tensor

    def findBiggest(self, cls=None):
        return self._find_by_area(cls=cls, mode="max")

    def findSmallest(self, cls=None):
        return self._find_by_area(cls=cls, mode="min")

    def findSmalles(self, cls=None):
        return self.findSmallest(cls=cls)

    def findAverage(self, cls=None):
        items = self._filtered(cls)
        if not items:
            return None
        areas = np.asarray([b.xywh[2] * b.xywh[3] for b in items], dtype=np.float32)
        target = float(areas.mean())
        return min(items, key=lambda b: abs((b.xywh[2] * b.xywh[3]) - target))

    def _filtered(self, cls=None):
        items = list(iter(self))
        if cls is None:
            return items
        if isinstance(cls, str):
            return [b for b in items if b.cls == cls]
        return [b for b in items if b.cls_id == int(cls)]

    def _find_by_area(self, cls=None, mode="max"):
        items = self._filtered(cls)
        if not items:
            return None
        key = lambda b: b.xywh[2] * b.xywh[3]
        return max(items, key=key) if mode == "max" else min(items, key=key)


class Results:
    def __init__(self, orig_img, boxes, scores, classes, names=None):
        self.orig_img = orig_img
        self.names = names or {}
        self.scores = scores.detach().cpu() if torch.is_tensor(scores) else torch.as_tensor(scores)
        self.classes = classes.detach().cpu() if torch.is_tensor(classes) else torch.as_tensor(classes)
        self.image_size = self._image_size(orig_img)
        self.boxes = DetectionBoxes(boxes, self.scores, self.classes, self.names, self.image_size)
        self.boxes_tensor = self.boxes.tensor

    @staticmethod
    def _image_size(img):
        if isinstance(img, Image.Image):
            return img.size
        h, w = getattr(img, "shape", (640, 640))[:2]
        return int(w), int(h)

    def _boxes_xyxy_px(self):
        img_w, img_h = self.image_size
        boxes = self.boxes.numpy()
        if len(boxes) == 0:
            return np.zeros((0, 4))
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (x - w / 2) * img_w
        y1 = (y - h / 2) * img_h
        x2 = (x + w / 2) * img_w
        y2 = (y + h / 2) * img_h
        return np.stack([x1, y1, x2, y2], axis=1)

    def plot(self, show_conf=True, show_labels=True):
        img = np.array(self.orig_img) if isinstance(self.orig_img, Image.Image) else self.orig_img.copy()
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        try:
            import cv2
            xyxy = self._boxes_xyxy_px()
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
                sc = float(self.scores[i])
                cl = int(self.classes[i])
                label = self.names.get(cl, str(cl))

                cv2.rectangle(img, c1, c2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                if show_labels:
                    txt = f"{label} {sc:.2f}" if show_conf else label
                    tf = 1
                    t_size = cv2.getTextSize(txt, 0, fontScale=2 / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(img, c1, c2, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(img, txt, (c1[0], c1[1] - 2), 0, 2 / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            return img
        except ImportError:
            if _HAS_MPL:
                fig, ax = plt.subplots(1)
                ax.imshow(img)
                xyxy = self._boxes_xyxy_px()
                for x1, y1, x2, y2 in xyxy:
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none")
                    ax.add_patch(rect)
                plt.axis("off")
                fig.canvas.draw()
                plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                return plot_img
            return img

    def show(self, **kwargs):
        plot_img = self.plot(**kwargs)
        if _HAS_MPL:
            plt.imshow(plot_img)
            plt.axis("off")
            plt.show()
        else:
            try:
                import cv2
                cv2.imshow("FOTO-NET Detection", cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except ImportError:
                print("No visualization backend (matplotlib/opencv) found.")

    def save(self, filename, **kwargs):
        plot_img = self.plot(**kwargs)
        Image.fromarray(plot_img).save(filename)

    def to_json(self):
        xyxy = self._boxes_xyxy_px()
        out = []
        for i in range(len(self.boxes)):
            cl = int(self.classes[i])
            out.append({
                "class": cl,
                "name": self.names.get(cl, str(cl)),
                "confidence": float(self.scores[i]),
                "box": xyxy[i].tolist(),
            })
        return json.dumps(out)

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_pandas()")
        xyxy = self._boxes_xyxy_px()
        data = []
        for i in range(len(self.boxes)):
            cl = int(self.classes[i])
            data.append({
                "class": cl,
                "name": self.names.get(cl, str(cl)),
                "confidence": float(self.scores[i]),
                "x1": xyxy[i, 0],
                "y1": xyxy[i, 1],
                "x2": xyxy[i, 2],
                "y2": xyxy[i, 3],
            })
        return pd.DataFrame(data)

    def __repr__(self):
        return f"FOTONET Results: {len(self.boxes)} detections"
