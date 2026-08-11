"""Render the FOTO-NET transform-layer motion study as a review MP4.

This is intentionally a local review artifact. The generated file lives under
``outputs/`` (gitignored) and is not part of the published website until the
motion direction is approved.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 30

INK = "#15271d"
FOREST = "#174d37"
DEEP = "#0d3828"
MOSS = "#5e806e"
MINT = "#b9d9bd"
PAPER = "#f3eddf"
PAPER_2 = "#e9dfcc"
RUST = "#c96543"
GOLD = "#d5a53f"
TEAL = "#2e7b79"
BERRY = "#9e4c66"
WHITE = "#fffaf0"
MUTED = "#6c756c"

FONT_DIR = Path("C:/Windows/Fonts")
SERIF = FONT_DIR / "georgia.ttf"
SERIF_BOLD = FONT_DIR / "georgiab.ttf"
SANS = FONT_DIR / "segoeui.ttf"
SANS_BOLD = FONT_DIR / "segoeuib.ttf"
MONO = FONT_DIR / "consola.ttf"
MONO_BOLD = FONT_DIR / "consolab.ttf"


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


F = {
    "display": font(SERIF, 78),
    "display_sm": font(SERIF, 62),
    "serif": font(SERIF, 38),
    "serif_b": font(SERIF_BOLD, 38),
    "h2": font(SANS_BOLD, 34),
    "h3": font(SANS_BOLD, 25),
    "body": font(SANS, 25),
    "body_b": font(SANS_BOLD, 25),
    "small": font(SANS, 19),
    "small_b": font(SANS_BOLD, 19),
    "mono": font(MONO, 19),
    "mono_b": font(MONO_BOLD, 19),
    "micro": font(MONO_BOLD, 15),
}


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def smooth(v: float) -> float:
    v = clamp(v)
    return v * v * (3.0 - 2.0 * v)


def smoother(v: float) -> float:
    v = clamp(v)
    return v * v * v * (v * (v * 6.0 - 15.0) + 10.0)


def out_back(v: float) -> float:
    v = clamp(v) - 1.0
    return 1.0 + 2.70158 * v**3 + 1.70158 * v**2


def pulse(t: float, rate: float = 1.0) -> float:
    return 0.5 + 0.5 * math.sin(t * math.tau * rate)


def mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix_box(a: tuple[float, ...], b: tuple[float, ...], t: float) -> tuple[int, ...]:
    return tuple(round(mix(x, y, t)) for x, y in zip(a, b))


def alpha(hex_color: str, opacity: int) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4)) + (opacity,)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, *, f: str = "body", fill: str = INK,
         anchor: str | None = None) -> None:
    draw.text(xy, value, font=F[f], fill=fill, anchor=anchor)


def round_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill: str | tuple,
               outline: str | tuple | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def backdrop(dark: bool = False) -> Image.Image:
    bg = DEEP if dark else PAPER
    im = Image.new("RGB", (WIDTH, HEIGHT), bg)
    d = ImageDraw.Draw(im, "RGBA")
    grid = alpha(MINT if dark else FOREST, 12 if dark else 10)
    for x in range(48, WIDTH, 96):
        d.line((x, 0, x, HEIGHT), fill=grid, width=1)
    for y in range(54, HEIGHT, 96):
        d.line((0, y, WIDTH, y), fill=grid, width=1)
    # Small, deterministic print-like flecks add life without reducing contrast.
    for i in range(84):
        x = (i * 223 + 71) % WIDTH
        y = (i * 137 + 43) % HEIGHT
        r = 1 + (i % 3 == 0)
        d.ellipse((x - r, y - r, x + r, y + r), fill=alpha(MINT if dark else FOREST, 18))
    return im


def header(im: Image.Image, chapter: str, progress: float, dark: bool = False) -> None:
    d = ImageDraw.Draw(im, "RGBA")
    fg = WHITE if dark else INK
    muted = MINT if dark else MUTED
    text(d, (80, 58), "F:", f="serif_b", fill=fg)
    text(d, (132, 65), "fotonet", f="h3", fill=fg)
    text(d, (WIDTH - 80, 70), chapter.upper(), f="micro", fill=muted, anchor="ra")
    d.line((80, 118, WIDTH - 80, 118), fill=alpha(MINT if dark else FOREST, 60), width=2)
    d.line((80, 118, 80 + int((WIDTH - 160) * clamp(progress)), 118), fill=RUST if not dark else GOLD, width=4)


def footer(im: Image.Image, left: str, dark: bool = False) -> None:
    d = ImageDraw.Draw(im, "RGBA")
    fg = MINT if dark else MUTED
    text(d, (80, HEIGHT - 46), left.upper(), f="micro", fill=fg)
    text(d, (WIDTH - 80, HEIGHT - 46), "GEOMETRY YOU CAN USE", f="micro", fill=fg, anchor="ra")


def intro_motion(d: ImageDraw.ImageDraw, t: float) -> None:
    e = out_back(t / 1.3)
    # Geometric pieces settle once, mirroring the site's quiet decorative language.
    shapes = [
        ((-90, 220, 340, 650), MOSS, (-40, 270)),
        ((1510, 150, 2020, 530), TEAL, (1590, 220)),
        ((1470, 690, 1900, 1120), RUST, (1530, 750)),
    ]
    for (x1, y1, x2, y2), color, (tx, ty) in shapes:
        dx = mix(-200 if x1 < 500 else 240, 0, e)
        dy = mix(130, 0, e)
        d.rounded_rectangle((x1 + dx, y1 + dy, x2 + dx, y2 + dy), radius=150, fill=alpha(color, 155))


def scene_intro(t: float, dur: float, progress: float) -> Image.Image:
    im = backdrop(dark=True)
    d = ImageDraw.Draw(im, "RGBA")
    intro_motion(d, t)
    header(im, "00 / motion study", progress, dark=True)
    rise = 34 * (1 - smooth(t / 1.0))
    text(d, (210, 328 + rise), "The transform layer,", f="display", fill=WHITE)
    text(d, (210, 420 + rise), "seen—not merely described.", f="display", fill=WHITE)
    text(d, (215, 565), "DETECTIONS  →  APPLICATION GEOMETRY", f="mono_b", fill=GOLD)
    text(d, (215, 636), "Anchor-aware crops, regions, gates, and relationships", f="body", fill=MINT)
    round_rect(d, (215, 720, 606, 780), 30, alpha(WHITE, 16), outline=alpha(MINT, 80), width=2)
    text(d, (410, 750), "REVIEW REEL · NO AP CLAIMS", f="micro", fill=WHITE, anchor="mm")
    footer(im, "01 · all motion is deterministic", dark=True)
    return im


ANCHORS = {
    "CENTER": (0.5, 0.5),
    "TOP": (0.5, 0.0),
    "BOTTOM": (0.5, 1.0),
    "TOP LEFT": (0.0, 0.0),
    "BOTTOM RIGHT": (1.0, 1.0),
}


def anchored_box(anchor: tuple[float, float], size: tuple[float, float], a: tuple[float, float]) -> tuple[int, int, int, int]:
    ax, ay = a
    w, h = size
    x, y = anchor
    return round(x - w * ax), round(y - h * ay), round(x + w * (1 - ax)), round(y + h * (1 - ay))


def draw_anchor(d: ImageDraw.ImageDraw, xy: tuple[float, float], t: float, color: str = RUST) -> None:
    x, y = xy
    ring = 18 + 9 * pulse(t, 1.3)
    d.ellipse((x - ring, y - ring, x + ring, y + ring), outline=alpha(color, 90), width=4)
    d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=WHITE)


def draw_person(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], opacity: int = 255) -> None:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    w, h = x2 - x1, y2 - y1
    d.ellipse((cx - w * .12, y1 + h * .12, cx + w * .12, y1 + h * .36), fill=alpha(FOREST, opacity))
    d.rounded_rectangle((cx - w * .2, y1 + h * .36, cx + w * .2, y1 + h * .76), radius=18, fill=alpha(MOSS, opacity))
    d.line((cx - w * .08, y1 + h * .72, cx - w * .18, y2 - h * .07), fill=alpha(INK, opacity), width=max(5, int(w * .055)))
    d.line((cx + w * .08, y1 + h * .72, cx + w * .18, y2 - h * .07), fill=alpha(INK, opacity), width=max(5, int(w * .055)))


def scene_anchors(t: float, dur: float, progress: float) -> Image.Image:
    im = backdrop()
    d = ImageDraw.Draw(im, "RGBA")
    header(im, "01 / anchor behavior", progress)
    text(d, (110, 180), "Choose what stays fixed.", f="display_sm")
    text(d, (115, 265), "The same expansion means different geometry when its anchor changes.", f="body", fill=MUTED)

    card = (100, 350, 1820, 950)
    round_rect(d, card, 58, WHITE, outline=alpha(FOREST, 45), width=2)
    names = list(ANCHORS)
    segment = dur / len(names)
    idx = min(len(names) - 1, int(t / segment))
    local = (t - idx * segment) / segment
    name = names[idx]
    a = ANCHORS[name]
    fixed = (960, 680)
    # Keep every directional expansion inside the demonstration card. A shared
    # fixed coordinate is more important here than a theatrically large box.
    small = anchored_box(fixed, (210, 190), a)
    large = anchored_box(fixed, (430, 300), a)
    grow = smoother(clamp(local / .68))
    box = mix_box(small, large, grow)
    draw_person(d, small, opacity=72)
    d.rounded_rectangle(box, radius=26, outline=FOREST, width=6)
    # Corner ticks make this read as precise geometry rather than generic UI.
    x1, y1, x2, y2 = box
    tick = 32
    for x, sx in ((x1, 1), (x2, -1)):
        for y, sy in ((y1, 1), (y2, -1)):
            d.line((x, y, x + tick * sx, y), fill=RUST, width=7)
            d.line((x, y, x, y + tick * sy), fill=RUST, width=7)
    draw_anchor(d, fixed, t)
    round_rect(d, (125, 390, 420, 458), 32, PAPER_2)
    text(d, (272, 424), name, f="mono_b", fill=FOREST, anchor="mm")
    text(d, (135, 520), "fixed coordinate", f="small_b", fill=INK)
    text(d, (135, 558), f"({int(fixed[0])}, {int(fixed[1])})", f="mono", fill=RUST)
    text(d, (135, 650), "box.scale(1.8,", f="mono", fill=MUTED)
    text(d, (135, 684), f"  anchor={name.replace(' ', '_')})", f="mono_b", fill=FOREST)
    # Navigation dots double as a chapter-time indicator.
    for i, label in enumerate(names):
        x = 585 + i * 240
        active = i == idx
        d.ellipse((x - 7, 893 - 7, x + 7, 893 + 7), fill=RUST if active else alpha(FOREST, 60))
        text(d, (x + 18, 893), label, f="micro", fill=INK if active else MUTED, anchor="lm")
    footer(im, "02 · AnchorPoint")
    return im


def scene_focus(t: float, dur: float, progress: float) -> Image.Image:
    im = backdrop(dark=True)
    d = ImageDraw.Draw(im, "RGBA")
    header(im, "02 / focus region", progress, dark=True)
    text(d, (110, 178), "Aim the geometry at what matters.", f="display_sm", fill=WHITE)
    text(d, (115, 264), "A FocusRegion turns a full detection into a semantic sub-region.", f="body", fill=MINT)
    panel = (104, 348, 1250, 943)
    round_rect(d, panel, 58, alpha(WHITE, 14), outline=alpha(MINT, 65), width=2)

    subject = (510, 405, 820, 900)
    draw_person(d, subject, 235)
    d.rounded_rectangle(subject, radius=24, outline=alpha(MINT, 150), width=4)
    # The highlighted top 55% arrives with an intentional scan motion.
    p = smoother(clamp((t - .6) / 1.4))
    focus_bottom = round(mix(subject[1], subject[1] + (subject[3] - subject[1]) * .55, p))
    focus_box = (subject[0], subject[1], subject[2], focus_bottom)
    if p > 0:
        d.rounded_rectangle(focus_box, radius=22, fill=alpha(GOLD, 55), outline=GOLD, width=6)
        d.line((subject[0] - 24, focus_bottom, subject[2] + 24, focus_bottom), fill=alpha(GOLD, 180), width=3)
    draw_anchor(d, ((subject[0] + subject[2]) / 2, focus_bottom), t, GOLD)

    round_rect(d, (1325, 392, 1784, 837), 44, PAPER)
    text(d, (1370, 438), "upper_body", f="h2", fill=FOREST)
    text(d, (1370, 506), "x: 0.00 → 1.00", f="mono", fill=MUTED)
    text(d, (1370, 547), "y: 0.00 → 0.55", f="mono", fill=MUTED)
    d.line((1370, 590, 1738, 590), fill=alpha(FOREST, 60), width=2)
    text(d, (1370, 638), "USE CASE", f="micro", fill=RUST)
    text(d, (1370, 680), "Keep a face-sized crop", f="body_b")
    text(d, (1370, 718), "tied to a person box", f="body_b")
    round_rect(d, (1370, 762, 1708, 811), 24, FOREST)
    text(d, (1539, 786), "detection.focus(region)", f="micro", fill=WHITE, anchor="mm")
    footer(im, "03 · FocusRegion", dark=True)
    return im


def scene_pipeline(t: float, dur: float, progress: float) -> Image.Image:
    im = backdrop()
    d = ImageDraw.Draw(im, "RGBA")
    header(im, "03 / crop composition", progress)
    text(d, (110, 174), "A crop is a sequence, not a guess.", f="display_sm")
    text(d, (115, 260), "Each operation is visible, ordered, and anchor-preserving.", f="body", fill=MUTED)

    steps = [
        ("DETECTION", (665, 455, 925, 868), FOREST),
        ("FOCUS", (665, 455, 925, 682), GOLD),
        ("PAD", (625, 420, 965, 718), RUST),
        ("4:5 ASPECT", (575, 420, 1015, 772), TEAL),
        ("RESTORE ANCHOR", (575, 516, 1015, 868), BERRY),
        ("CLAMP", (575, 516, 1015, 868), FOREST),
        ("CROP", (575, 516, 1015, 868), RUST),
    ]
    segment = dur / len(steps)
    idx = min(len(steps) - 1, int(t / segment))
    local = (t - idx * segment) / segment
    prev = steps[max(0, idx - 1)][1] if idx else steps[0][1]
    label, target, color = steps[idx]
    box = mix_box(prev, target, smoother(clamp(local / .66)))

    round_rect(d, (120, 350, 1240, 930), 56, WHITE, outline=alpha(FOREST, 50), width=2)
    # Image boundary and subject.
    d.rounded_rectangle((430, 392, 1165, 895), radius=36, fill=alpha(MOSS, 26), outline=alpha(MOSS, 80), width=2)
    draw_person(d, (665, 455, 925, 868), 145)
    d.rounded_rectangle(box, radius=24, outline=color, width=7)
    if label == "CROP":
        veil = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        vd = ImageDraw.Draw(veil)
        vd.rectangle((430, 392, 1165, box[1]), fill=alpha(INK, 75))
        vd.rectangle((430, box[3], 1165, 895), fill=alpha(INK, 75))
        vd.rectangle((430, box[1], box[0], box[3]), fill=alpha(INK, 75))
        vd.rectangle((box[2], box[1], 1165, box[3]), fill=alpha(INK, 75))
        im.alpha_composite(veil) if im.mode == "RGBA" else im.paste(Image.alpha_composite(im.convert("RGBA"), veil).convert("RGB"))
        d = ImageDraw.Draw(im, "RGBA")
    anchor_y = box[3] if idx >= 4 else box[1]
    draw_anchor(d, ((box[0] + box[2]) / 2, anchor_y), t, color)
    round_rect(d, (152, 390, 382, 454), 30, color)
    text(d, (267, 422), label, f="micro", fill=WHITE, anchor="mm")

    # The right rail exposes exact operation order.
    round_rect(d, (1300, 350, 1810, 930), 52, DEEP)
    for i, (name, _, step_color) in enumerate(steps):
        y = 405 + i * 70
        done = i < idx
        active = i == idx
        d.ellipse((1350, y - 9, 1368, y + 9), fill=step_color if active else (MINT if done else alpha(MINT, 45)))
        if i < len(steps) - 1:
            d.line((1359, y + 15, 1359, y + 55), fill=alpha(MINT, 65), width=2)
        text(d, (1400, y), name, f="small_b" if active else "small", fill=WHITE if active else MINT, anchor="lm")
    text(d, (1350, 883), "safe_crop(image, box)", f="mono", fill=GOLD)
    footer(im, "04 · BoxTransform")
    return im


def scene_zone(t: float, dur: float, progress: float) -> Image.Image:
    im = backdrop(dark=True)
    d = ImageDraw.Draw(im, "RGBA")
    header(im, "04 / zone events", progress, dark=True)
    text(d, (110, 174), "Events from geometry, not vibes.", f="display_sm", fill=WHITE)
    text(d, (115, 260), "A bottom anchor crossing a region becomes a stable enter / exit signal.", f="body", fill=MINT)

    round_rect(d, (110, 350, 1340, 930), 58, alpha(WHITE, 13), outline=alpha(MINT, 62), width=2)
    zone = (590, 530, 1190, 855)
    d.rounded_rectangle(zone, radius=42, fill=alpha(TEAL, 58), outline=TEAL, width=5)
    text(d, (620, 570), "RESTRICTED ZONE", f="micro", fill=MINT)
    # Subject travels across the zone; state is computed from bottom anchor.
    travel = 0.5 - 0.5 * math.cos((t / dur) * math.tau)
    cx = mix(280, 1510, travel)
    subject = (int(cx - 75), 515, int(cx + 75), 820)
    draw_person(d, subject, 235)
    d.rounded_rectangle(subject, radius=20, outline=WHITE, width=4)
    anchor_xy = (cx, subject[3])
    inside = zone[0] <= anchor_xy[0] <= zone[2] and zone[1] <= anchor_xy[1] <= zone[3]
    draw_anchor(d, anchor_xy, t, GOLD if inside else RUST)
    # Motion trail.
    d.line((280, subject[3], 1510, subject[3]), fill=alpha(MINT, 55), width=3)

    round_rect(d, (1410, 390, 1810, 830), 45, PAPER)
    text(d, (1450, 438), "bottom anchor", f="h2", fill=FOREST)
    text(d, (1450, 500), "inside(region)", f="mono", fill=MUTED)
    pill = (1450, 555, 1768, 625)
    round_rect(d, pill, 35, FOREST if inside else PAPER_2, outline=FOREST, width=2)
    text(d, ((pill[0] + pill[2]) // 2, (pill[1] + pill[3]) // 2), "TRUE · ENTER" if inside else "FALSE · OUTSIDE",
         f="small_b", fill=WHITE if inside else FOREST, anchor="mm")
    text(d, (1450, 688), "event payload", f="micro", fill=RUST)
    text(d, (1450, 730), '{"track": 17,', f="mono", fill=INK)
    text(d, (1450, 764), f' "state": "{"enter" if inside else "exit"}"}}', f="mono_b", fill=TEAL if inside else RUST)
    footer(im, "05 · AnchorPoint.inside", dark=True)
    return im


def scene_containment(t: float, dur: float, progress: float) -> Image.Image:
    im = backdrop()
    d = ImageDraw.Draw(im, "RGBA")
    header(im, "05 / containment policy", progress)
    text(d, (110, 174), "Point policy ≠ area policy.", f="display_sm")
    text(d, (115, 260), "Choose the rule that matches the event you actually mean.", f="body", fill=MUTED)

    cards = [(100, 350, 930, 925), (990, 350, 1820, 925)]
    for box in cards:
        round_rect(d, box, 54, WHITE, outline=alpha(FOREST, 50), width=2)
    text(d, (150, 405), "A · BOTTOM ANCHOR", f="micro", fill=RUST)
    text(d, (1040, 405), "B · BOX AREA", f="micro", fill=TEAL)
    text(d, (150, 450), "One meaningful point", f="h2")
    text(d, (1040, 450), "75% must overlap", f="h2")

    phase = smooth((math.sin(t * 1.15) + 1) / 2)
    for ci, xoff in enumerate((0, 890)):
        region = (300 + xoff, 590, 735 + xoff, 840)
        d.rounded_rectangle(region, radius=32, fill=alpha(MINT, 92), outline=FOREST, width=4)
        bx = int(mix(210 + xoff, 510 + xoff, phase))
        obj = (bx, 505, bx + 210, 790)
        draw_person(d, obj, 130)
        d.rounded_rectangle(obj, radius=22, outline=RUST if ci == 0 else TEAL, width=5)
        overlap_x = max(0, min(obj[2], region[2]) - max(obj[0], region[0]))
        overlap_y = max(0, min(obj[3], region[3]) - max(obj[1], region[1]))
        ratio = overlap_x * overlap_y / ((obj[2] - obj[0]) * (obj[3] - obj[1]))
        if ci == 0:
            ap = ((obj[0] + obj[2]) / 2, obj[3])
            valid = region[0] <= ap[0] <= region[2] and region[1] <= ap[1] <= region[3]
            draw_anchor(d, ap, t, RUST)
            status = "INSIDE" if valid else "OUTSIDE"
        else:
            valid = ratio >= .75
            if overlap_x and overlap_y:
                d.rectangle((max(obj[0], region[0]), max(obj[1], region[1]), min(obj[2], region[2]), min(obj[3], region[3])), fill=alpha(TEAL, 80))
            status = f"{ratio * 100:02.0f}% · {'PASS' if valid else 'WAIT'}"
        round_rect(d, (320 + xoff, 852, 715 + xoff, 897), 22, FOREST if valid else PAPER_2)
        text(d, (517 + xoff, 874), status, f="micro", fill=WHITE if valid else MUTED, anchor="mm")
    footer(im, "06 · explicit containment semantics")
    return im


def scene_relationships(t: float, dur: float, progress: float) -> Image.Image:
    im = backdrop(dark=True)
    d = ImageDraw.Draw(im, "RGBA")
    header(im, "06 / spatial relationships", progress, dark=True)
    text(d, (110, 174), "Measure relationships directly.", f="display_sm", fill=WHITE)
    text(d, (115, 260), "Intersection-over-union and anchor distance answer different questions.", f="body", fill=MINT)

    round_rect(d, (105, 350, 1815, 920), 58, alpha(WHITE, 13), outline=alpha(MINT, 65), width=2)
    mid = dur * .52
    first = smooth(clamp(t / mid))
    second = smooth(clamp((t - mid + .5) / (dur - mid)))
    a = (430, 495, 810, 800)
    bx = int(mix(1160, 690, first))
    b = (bx, 555, bx + 380, 860)
    d.rounded_rectangle(a, radius=34, fill=alpha(GOLD, 34), outline=GOLD, width=6)
    d.rounded_rectangle(b, radius=34, fill=alpha(TEAL, 40), outline=TEAL, width=6)
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    iou = intersection / union
    if intersection:
        d.rounded_rectangle((ix1, iy1, ix2, iy2), radius=18, fill=alpha(RUST, 125))

    pa = ((a[0] + a[2]) / 2, a[3])
    pb = ((b[0] + b[2]) / 2, b[3])
    if second > .05:
        d.line((*pa, *pb), fill=alpha(WHITE, int(180 * second)), width=4)
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        distance = math.dist(pa, pb)
        round_rect(d, (int(mx - 115), int(my - 36), int(mx + 115), int(my + 36)), 28, PAPER)
        text(d, (int(mx), int(my)), f"{distance:0.0f} px", f="mono_b", fill=FOREST, anchor="mm")
        draw_anchor(d, pa, t, GOLD)
        draw_anchor(d, pb, t, TEAL)

    round_rect(d, (1290, 440, 1740, 788), 42, PAPER)
    text(d, (1340, 486), "RELATIONSHIP", f="micro", fill=RUST)
    text(d, (1340, 544), f"IoU  {iou:0.3f}", f="h2", fill=FOREST)
    d.rectangle((1340, 606, 1688, 632), fill=PAPER_2)
    d.rectangle((1340, 606, 1340 + int(348 * min(1, iou * 2.4)), 632), fill=RUST)
    text(d, (1340, 694), "bottom-anchor distance", f="small", fill=MUTED)
    text(d, (1340, 738), f"{math.dist(pa, pb):0.1f} px", f="mono_b", fill=TEAL)
    footer(im, "07 · overlap and distance", dark=True)
    return im


def scene_outro(t: float, dur: float, progress: float) -> Image.Image:
    im = backdrop()
    d = ImageDraw.Draw(im, "RGBA")
    header(im, "07 / proposed motion system", progress)
    intro_motion(d, t)
    text(d, (210, 330), "Detections become", f="display", fill=INK)
    text(d, (210, 420), "useful application geometry.", f="display", fill=INK)
    items = [
        ("ANCHOR", RUST), ("FOCUS", GOLD), ("TRANSFORM", TEAL),
        ("CONTAIN", BERRY), ("RELATE", FOREST),
    ]
    for i, (label, color) in enumerate(items):
        x = 220 + i * 310
        delay = .5 + i * .12
        y = int(mix(655, 605, out_back((t - delay) / .8)))
        round_rect(d, (x, y, x + 260, y + 82), 38, color)
        text(d, (x + 130, y + 41), label, f="micro", fill=WHITE, anchor="mm")
    text(d, (215, 775), "MOTION STUDY 01", f="mono_b", fill=FOREST)
    text(d, (215, 820), "Ready for design review · not yet published", f="body", fill=MUTED)
    footer(im, "08 · fotonet transform layer")
    return im


SCENES = [
    ("intro", 4.5, scene_intro),
    ("anchors", 7.5, scene_anchors),
    ("focus", 6.0, scene_focus),
    ("pipeline", 8.0, scene_pipeline),
    ("zone", 7.0, scene_zone),
    ("containment", 6.5, scene_containment),
    ("relationships", 7.0, scene_relationships),
    ("outro", 4.5, scene_outro),
]


def frame_with_fade(frame: Image.Image, local_t: float, duration: float, dark: bool) -> Image.Image:
    fade_in = smooth(local_t / .34)
    fade_out = smooth((duration - local_t) / .34)
    visibility = min(fade_in, fade_out)
    if visibility >= .999:
        return frame
    blank = backdrop(dark=dark)
    return Image.blend(blank, frame, visibility)


def render(output: Path, fps: int, width: int, height: int) -> None:
    if width != WIDTH or height != HEIGHT:
        raise ValueError(f"This study is art-directed at {WIDTH}x{HEIGHT}; got {width}x{height}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer with the mp4v codec.")

    total_duration = sum(item[1] for item in SCENES)
    total_frames = round(total_duration * fps)
    scene_start = 0.0
    frame_index = 0
    posters: list[Image.Image] = []
    for scene_index, (name, duration, painter) in enumerate(SCENES):
        count = round(duration * fps)
        for i in range(count):
            local_t = i / fps
            absolute_t = scene_start + local_t
            progress = absolute_t / total_duration
            frame = painter(local_t, duration, progress)
            dark = scene_index in (0, 2, 4, 6)
            frame = frame_with_fade(frame, local_t, duration, dark)
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            if i == count // 2:
                posters.append(frame.copy())
            frame_index += 1
            if frame_index % (fps * 5) == 0:
                print(f"rendered {frame_index}/{total_frames} frames")
        scene_start += duration
    writer.release()

    # A companion contact sheet makes visual QA fast without replacing video review.
    thumb_w, thumb_h = 640, 360
    sheet = Image.new("RGB", (thumb_w * 2, thumb_h * 4), PAPER)
    for index, poster in enumerate(posters):
        poster.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(poster, ((index % 2) * thumb_w, (index // 2) * thumb_h))
    sheet.save(output.with_suffix(".contact-sheet.jpg"), quality=92, optimize=True)
    print(f"wrote {output} ({frame_index} frames, {frame_index / fps:.2f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "fotonet-transform-motion-preview.mp4",
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(args.output.resolve(), args.fps, args.width, args.height)
