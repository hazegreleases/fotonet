"""Google AGY-style terminal UI, responsive progress bars, and diagnostics for Fotonet.

Features:
- Google AGY aesthetic: braille spinner, clean line progress bar, refined color palette.
- Independent, constant ~12 FPS braille spinner animation on a background thread.
- Dynamic terminal width adaptation: never wraps, clips, or breaks lines when terminal resizes.
- Compact, high-signal information density with strictly zero emojis.
- Cross-platform: Linux, macOS, and Windows (with virtual terminal processing support).
- Automatic non-TTY fallback for CI, headless runs, and file redirection.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import sys
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# Enable Windows UTF-8 and virtual terminal processing if available
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle and handle != -1:
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _supports_color() -> bool:
    """Check whether colors should be enabled."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if not sys.stdout.isatty() and not os.environ.get("FORCE_COLOR"):
        return False
    return True


def _supports_unicode() -> bool:
    """Check if the terminal stream supports UTF-8 Unicode characters."""
    try:
        encoding = (sys.stdout.encoding or "utf-8").lower()
        return "utf" in encoding
    except Exception:
        return False


COLOR_ENABLED = _supports_color()
UNICODE_ENABLED = _supports_unicode()


class Ansi:
    """ANSI color & styling helper with automatic enable/disable."""
    RESET = "\033[0m" if COLOR_ENABLED else ""
    BOLD = "\033[1m" if COLOR_ENABLED else ""
    DIM = "\033[2m" if COLOR_ENABLED else ""
    ITALIC = "\033[3m" if COLOR_ENABLED else ""
    UNDERLINE = "\033[4m" if COLOR_ENABLED else ""

    # AGY Palette
    BLUE = "\033[38;5;39m" if COLOR_ENABLED else ""      # Google blue/cyan
    CYAN = "\033[38;5;51m" if COLOR_ENABLED else ""      # Bright cyan
    MINT = "\033[38;5;48m" if COLOR_ENABLED else ""      # Soft green / success
    GREEN = "\033[38;5;42m" if COLOR_ENABLED else ""     # Vibrant green
    AMBER = "\033[38;5;214m" if COLOR_ENABLED else ""    # Warning / highlight
    ORANGE = "\033[38;5;208m" if COLOR_ENABLED else ""   # Attention
    RED = "\033[38;5;203m" if COLOR_ENABLED else ""      # Critical / error
    PURPLE = "\033[38;5;141m" if COLOR_ENABLED else ""   # Dimensions / tags
    GRAY = "\033[38;5;245m" if COLOR_ENABLED else ""     # Subtle text
    DARK_GRAY = "\033[38;5;239m" if COLOR_ENABLED else ""# Borders / dividers
    WHITE = "\033[1;97m" if COLOR_ENABLED else ""        # Emphasized text

    # Clear codes
    CLEAR_LINE = "\033[2K"
    CLEAR_TO_END = "\033[K"

    @classmethod
    def paint(cls, text: str, *styles: str) -> str:
        if not COLOR_ENABLED or not styles:
            return text
        return f"{''.join(styles)}{text}{cls.RESET}"


# Regex to strip ANSI sequences for visible length calculations
_ANSI_STRIP_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def visible_len(text: str) -> int:
    """Calculate the visible character length of a string, ignoring ANSI codes."""
    return len(_ANSI_STRIP_RE.sub("", text))


def fit_line(text: str, max_cols: int) -> str:
    """Truncate a styled line so its visible length does not exceed max_cols."""
    if max_cols <= 0:
        return ""
    cur_len = visible_len(text)
    if cur_len <= max_cols:
        return text

    ellipsis = "…" if UNICODE_ENABLED else "..."
    target_len = max(max_cols - len(ellipsis), 0)
    visible_count = 0
    out = []
    i = 0
    while i < len(text):
        match = _ANSI_STRIP_RE.match(text, i)
        if match:
            out.append(match.group(0))
            i = match.end()
            continue
        if visible_count >= target_len:
            break
        out.append(text[i])
        visible_count += 1
        i += 1

    out.append(ellipsis)
    if COLOR_ENABLED:
        out.append(Ansi.RESET)
    return "".join(out)


def get_terminal_width(default: int = 80) -> int:
    """Get the current terminal width in columns, clamped to a sensible minimum."""
    try:
        cols = shutil.get_terminal_size((default, 24)).columns
        return max(cols, 40)
    except Exception:
        return default


# AGY spinner frames
SPINNER_FRAMES = (
    ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    if UNICODE_ENABLED
    else ["-", "\\", "|", "/"]
)

# Strictly non-emoji minimal glyphs
GLYPHS = {
    "bullet": "*" if UNICODE_ENABLED else "*",
    "arrow": "->" if UNICODE_ENABLED else "->",
    "star": "*" if UNICODE_ENABLED else "*",
    "check": "+" if UNICODE_ENABLED else "+",
    "cross": "x" if UNICODE_ENABLED else "x",
    "bolt": "!" if UNICODE_ENABLED else "!",
    "info": "i" if UNICODE_ENABLED else "i",
    "trend_down": "v" if UNICODE_ENABLED else "v",
    "bar_fill": "━" if UNICODE_ENABLED else "=",
    "bar_tip": "╸" if UNICODE_ENABLED else ">",
    "bar_empty": "─" if UNICODE_ENABLED else "-",
    "box_tl": "╭" if UNICODE_ENABLED else "+",
    "box_tr": "╮" if UNICODE_ENABLED else "+",
    "box_bl": "╰" if UNICODE_ENABLED else "+",
    "box_br": "╯" if UNICODE_ENABLED else "+",
    "box_h": "─" if UNICODE_ENABLED else "-",
    "box_v": "│" if UNICODE_ENABLED else "|",
}


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable compact time string (e.g. 42s, 02m 15s, 01h 24m)."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes:02d}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}h {minutes:02d}m"


def format_eta(seconds: float) -> str:
    """Format ETA seconds into mm:ss or hh:mm:ss."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    total_sec = int(seconds)
    hours, rem = divmod(total_sec, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class AgyProgressBar:
    """A Google AGY-style responsive progress bar.

    - Independent, constant 12 FPS braille spinner animation on a background thread.
    - Adapts dynamically to terminal width on every tick without wrapping or clipping lines.
    - High-density telemetry and non-blocking update path.
    """

    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        total: Optional[int] = None,
        desc: str = "Training",
        epoch: Optional[int] = None,
        total_epochs: Optional[Any] = None,
        unit: str = "it",
        min_interval: float = 0.08,  # ~12 FPS refresh rate
        is_val: bool = False,
    ):
        self.iterable = iterable
        self.total = total or (len(iterable) if iterable is not None and hasattr(iterable, "__len__") else 0)
        self.desc = desc
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.unit = unit
        self.min_interval = min_interval
        self.is_val = is_val

        self.current = 0
        self.start_time = time.time()
        self.last_render_time = 0.0
        self.last_step_time = self.start_time
        self.rate_ema = 0.0
        self.is_tty = sys.stdout.isatty()
        self.closed = False
        self.metrics: Dict[str, Any] = {}
        # Whole-run context: when set, the bar leads with the run-level
        # fraction and shows a stable, externally computed ETA instead of an
        # epoch-local projection.
        self.run_done: int = 0
        self.run_total: int = 0
        self.run_eta_sec: Optional[float] = None
        self.run_elapsed_sec: float = 0.0
        self.phase: str = ""
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.is_tty:
            self._thread = threading.Thread(target=self._spinner_daemon, daemon=True)
            self._thread.start()

    def _spinner_daemon(self):
        """Active ~12 FPS background refresh ensuring the braille spinner spins smoothly."""
        while not self._stop_event.wait(self.min_interval):
            if self.closed:
                break
            with self._lock:
                if not self.closed:
                    self._do_render(time.time())

    def __iter__(self) -> Iterator:
        self.start_time = time.time()
        self.last_render_time = 0.0
        self.current = 0
        for item in self.iterable:
            yield item
            self.update(1)
        self.close()

    def update(self, n: int = 1):
        """Advance progress by n steps."""
        with self._lock:
            self.current += n
            now = time.time()
            elapsed = now - self.start_time
            if elapsed > 0.001 and self.current > 0:
                instant_rate = self.current / elapsed
                if self.rate_ema == 0.0:
                    self.rate_ema = instant_rate
                else:
                    self.rate_ema = 0.8 * self.rate_ema + 0.2 * instant_rate

            if self.current >= self.total:
                self._do_render(now)

    def set_postfix(self, **kwargs):
        """Update metrics to be displayed on the bar."""
        with self._lock:
            self.metrics.update(kwargs)

    def set_run(
        self,
        done: int,
        total: int,
        eta_sec: Optional[float] = None,
        elapsed_sec: float = 0.0,
    ):
        """Set whole-run progress plus a pre-computed, display-stable ETA.

        The ETA is computed by the caller (stage-aware projection); the bar
        only renders it, quantized to ten-minute ticks so the number never
        flickers.
        """
        with self._lock:
            self.run_done = max(int(done), 0)
            self.run_total = max(int(total), 0)
            self.run_eta_sec = eta_sec
            self.run_elapsed_sec = max(float(elapsed_sec), 0.0)

    def set_phase(self, phase: str):
        """Set the schedule-phase badge (e.g. WSD warmup/stable/decay)."""
        with self._lock:
            self.phase = str(phase or "")

    def _render_bar(self, width: int, progress: float) -> str:
        if width <= 0:
            return ""
        filled_len = int(round(width * progress))
        filled_len = max(0, min(width, filled_len))
        empty_len = width - filled_len

        if filled_len == 0:
            bar = GLYPHS["bar_empty"] * empty_len
            return Ansi.paint(bar, Ansi.DARK_GRAY)

        if empty_len == 0:
            bar = GLYPHS["bar_fill"] * filled_len
            return Ansi.paint(bar, Ansi.BLUE)

        bar_fill = GLYPHS["bar_fill"] * (filled_len - 1)
        bar_tip = GLYPHS["bar_tip"]
        bar_empty = GLYPHS["bar_empty"] * empty_len

        return (
            Ansi.paint(bar_fill, Ansi.BLUE)
            + Ansi.paint(bar_tip, Ansi.CYAN)
            + Ansi.paint(bar_empty, Ansi.DARK_GRAY)
        )

    def render(self, now: Optional[float] = None):
        with self._lock:
            self._do_render(now)

    def _do_render(self, now: Optional[float] = None):
        if not self.is_tty or self.closed:
            return

        if now is None:
            now = time.time()

        cols = get_terminal_width(80)
        max_width = cols - 1

        # Smooth, independent 10 FPS orbital spinner frame
        spinner_idx = int(now * 10) % len(SPINNER_FRAMES)
        spinner_char = SPINNER_FRAMES[spinner_idx]
        spinner_colored = Ansi.paint(spinner_char, Ansi.CYAN, Ansi.BOLD)

        total = self.total if self.total > 0 else 1
        epoch_fraction = min(max(self.current / total, 0.0), 1.0)

        phase_badge = ""
        if self.phase:
            phase_key = self.phase.split()[0].lower()
            phase_color = {
                "warmup": Ansi.CYAN,
                "stable": Ansi.BLUE,
                "decay": Ansi.AMBER,
            }.get(phase_key, Ansi.PURPLE)
            phase_badge = f" {Ansi.paint(f'[{self.phase}]', phase_color, Ansi.BOLD)}"

        if self.epoch is not None:
            total_ep = self.total_epochs if self.total_epochs is not None else "inf"
            prefix = f"{spinner_colored} {Ansi.paint(f'[{self.desc} {self.epoch}/{total_ep}]', Ansi.BOLD)}{phase_badge} "
        else:
            prefix = f"{spinner_colored} {Ansi.paint(f'[{self.desc}]', Ansi.BOLD)}{phase_badge} "

        elapsed_sec = now - self.start_time
        rate = self.rate_ema if self.rate_ema > 0 else (self.current / max(elapsed_sec, 1e-6))
        speed_str = f"{rate:.1f}{self.unit}/s" if rate < 100 else f"{int(rate)}{self.unit}/s"

        if self.run_total > 0:
            # Whole-run lead: the bar, percentage, elapsed, and ETA describe
            # the complete training run; the epoch counter stays visible.
            run_fraction = min(max(self.run_done / max(self.run_total, 1), 0.0), 1.0)
            fraction = run_fraction
            pct_str = f"{run_fraction * 100:5.1f}%"
            elapsed_str = format_eta(self.run_elapsed_sec)
            eta_sec = self.run_eta_sec if self.run_eta_sec is not None else 0.0
            # Quantize the displayed ETA to ten-minute ticks: the estimate is
            # only refreshed at epoch boundaries, and the tick floor keeps the
            # number monotone and calm between refreshes.
            eta_str = format_eta(math.floor(eta_sec / 600.0) * 600.0)
            time_stats = f"{elapsed_str}<-{eta_str}, {speed_str}"
            count_str = f"ep {epoch_fraction * 100:4.1f}% {self.current}/{self.total}"
        else:
            fraction = epoch_fraction
            pct_str = f"{int(fraction * 100):>3d}%"
            elapsed_str = format_eta(elapsed_sec)
            remaining_sec = (total - self.current) / rate if rate > 0 else 0
            eta_str = format_eta(remaining_sec)
            time_stats = f"{elapsed_str}<{eta_str}, {speed_str}"
            count_str = f"{self.current}/{self.total}"

        metrics_parts = []
        for k, v in self.metrics.items():
            if v is None:
                continue
            metrics_parts.append(f"{k} {Ansi.paint(str(v), Ansi.WHITE)}")
        metrics_str = " · ".join(metrics_parts)

        sep = Ansi.paint(" │ ", Ansi.DARK_GRAY)

        meta_right = f"{Ansi.paint(pct_str, Ansi.WHITE)} {Ansi.paint(count_str, Ansi.GRAY)} {Ansi.paint(f'[{time_stats}]', Ansi.DIM)}"
        if metrics_str:
            meta_right += f"{sep}{metrics_str}"

        fixed_len = visible_len(prefix) + visible_len(meta_right) + 1
        avail_bar = max_width - fixed_len

        if avail_bar >= 10:
            bar_str = self._render_bar(avail_bar, fraction)
            line = f"{prefix}{bar_str} {meta_right}"
        elif avail_bar >= 0:
            bar_str = self._render_bar(avail_bar, fraction)
            line = f"{prefix}{bar_str} {meta_right}"
        else:
            meta_short = f"{Ansi.paint(pct_str, Ansi.WHITE)} {Ansi.paint(count_str, Ansi.GRAY)}"
            avail_bar = max_width - visible_len(prefix) - visible_len(meta_short) - 1
            if avail_bar >= 6:
                bar_str = self._render_bar(avail_bar, fraction)
                line = f"{prefix}{bar_str} {meta_short}"
            else:
                line = f"{prefix} {meta_short}"

        sys.stdout.write(f"\r{fit_line(line, max_width)}{Ansi.CLEAR_TO_END}")
        sys.stdout.flush()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.15)
            except Exception:
                pass
        if self.is_tty:
            with self._lock:
                sys.stdout.write(f"\r{Ansi.CLEAR_TO_END}")
                sys.stdout.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def print_train_card(
    model_name: str,
    param_count: Optional[int],
    device: str,
    n_train: int,
    n_val: int,
    epochs: Any,
    imgsz: Any,
    batch_size: int,
    accum_steps: int,
    optimizer_name: str,
    lr0: float,
    scheduler_name: str,
    amp: bool,
    val_txt: str,
    resume_path: Optional[str] = None,
    recipe_name: Optional[str] = None,
):
    """Print an AGY-styled header card that dynamically formats rows without clipping."""
    term_width = get_terminal_width(80)

    if param_count:
        params_str = f"{param_count / 1_000_000:.2f}M" if param_count >= 1_000_000 else f"{param_count / 1_000:.1f}k"
    else:
        params_str = "auto"

    eff_batch = batch_size * max(accum_steps, 1)
    accum_str = f"{batch_size} (eff {eff_batch})" if accum_steps > 1 else f"{batch_size}"
    lr_str = f"{lr0:g}"

    row1_left = f"Model:    {Ansi.paint(model_name, Ansi.WHITE)} ({params_str} params)"
    row1_right = f"Hardware: {Ansi.paint(str(device), Ansi.WHITE)}"

    row2_left = f"Dataset:  {Ansi.paint(f'{n_train:,}', Ansi.WHITE)} train · {Ansi.paint(f'{n_val:,}', Ansi.WHITE)} val"
    row2_right = f"Batch:    {Ansi.paint(accum_str, Ansi.WHITE)}"

    sched_label = f"Schedule: {Ansi.paint(str(epochs), Ansi.WHITE)} epochs @ {Ansi.paint(str(imgsz) + 'px', Ansi.WHITE)}"
    policy_label = f"Policy:   {Ansi.paint(optimizer_name, Ansi.WHITE)} (lr {lr_str}, {scheduler_name})"

    # Calculate optimal dynamic width
    needed_for_pair = max(
        visible_len(row1_left) + visible_len(row1_right) + 6,
        visible_len(row2_left) + visible_len(row2_right) + 6,
    )
    needed_sched = visible_len(sched_label) + visible_len(policy_label) + 6
    can_dual_sched = (needed_sched <= term_width - 1)

    target_width = min(term_width - 1, max(needed_for_pair, needed_sched if can_dual_sched else 78, 80))

    # Title with ZERO emojis
    title = " FOTONET Training Engine "
    title_styled = Ansi.paint(title, Ansi.BOLD, Ansi.CYAN)
    title_len = visible_len(title)

    top_bar_right = GLYPHS["box_h"] * max(target_width - 2 - title_len, 2)
    top_line = f"{Ansi.paint(GLYPHS['box_tl'], Ansi.DARK_GRAY)}{title_styled}{Ansi.paint(top_bar_right + GLYPHS['box_tr'], Ansi.DARK_GRAY)}"
    bot_line = f"{Ansi.paint(GLYPHS['box_bl'] + GLYPHS['box_h'] * (target_width - 2) + GLYPHS['box_br'], Ansi.DARK_GRAY)}"

    def make_dual_row(left: str, right: str, width: int) -> str:
        inner_space = width - 4
        left_len = visible_len(left)
        right_len = visible_len(right)
        gap = max(inner_space - left_len - right_len, 2)
        inner = f" {left}{' ' * gap}{right} "
        actual_len = visible_len(inner)
        pad_end = max(width - 2 - actual_len, 0)
        return f"{Ansi.paint(GLYPHS['box_v'], Ansi.DARK_GRAY)}{inner}{' ' * pad_end}{Ansi.paint(GLYPHS['box_v'], Ansi.DARK_GRAY)}"

    def make_single_row(content: str, width: int) -> str:
        inner = f" {content} "
        actual_len = visible_len(inner)
        pad_end = max(width - 2 - actual_len, 0)
        return f"{Ansi.paint(GLYPHS['box_v'], Ansi.DARK_GRAY)}{inner}{' ' * pad_end}{Ansi.paint(GLYPHS['box_v'], Ansi.DARK_GRAY)}"

    print()
    print(top_line)
    print(make_dual_row(row1_left, row1_right, target_width))
    print(make_dual_row(row2_left, row2_right, target_width))

    if can_dual_sched and (visible_len(sched_label) + visible_len(policy_label) + 6 <= target_width):
        print(make_dual_row(sched_label, policy_label, target_width))
    else:
        print(make_single_row(sched_label, target_width))
        print(make_single_row(policy_label, target_width))

    if resume_path:
        resume_str = f"Resumed:  {Ansi.paint(str(resume_path), Ansi.AMBER)}"
        print(make_single_row(resume_str, target_width))
    if recipe_name:
        recipe_str = f"Recipe:   {Ansi.paint(str(recipe_name), Ansi.GRAY)}"
        print(make_single_row(recipe_str, target_width))

    print(bot_line)
    print()


def print_epoch_summary(
    epoch: int,
    total_epochs: Any,
    epoch_seconds: float,
    metrics: Dict[str, Any],
    val_metrics: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
    val_mode: str = "full",
    regression_name: str = "dfl",
    phase: str = "",
    run_progress: Optional[str] = None,
    run_eta: Optional[str] = None,
):
    """Print an AGY-styled compact epoch summary row with zero emojis."""
    if sys.stdout.isatty():
        sys.stdout.write(f"\r{Ansi.CLEAR_TO_END}")

    total_str = str(total_epochs)
    epoch_str = f"{epoch:>3d}/{total_str}"
    bullet = Ansi.paint("*", Ansi.CYAN)
    epoch_badge = f"{bullet} {Ansi.paint(f'Epoch {epoch_str}', Ansi.BOLD)}"
    if phase:
        phase_key = phase.split()[0].lower()
        phase_color = {
            "warmup": Ansi.CYAN,
            "stable": Ansi.BLUE,
            "decay": Ansi.AMBER,
        }.get(phase_key, Ansi.PURPLE)
        epoch_badge += f" {Ansi.paint(f'[{phase}]', phase_color, Ansi.BOLD)}"
    time_str = Ansi.paint(format_duration(epoch_seconds), Ansi.GRAY)

    loss_val = metrics.get("loss", 0.0)
    cls_val = metrics.get("cls_loss", metrics.get("cls", 0.0))
    reg_val = metrics.get(f"{regression_name}_loss", metrics.get("regpc", 0.0))
    iou_val = metrics.get("iou_loss", metrics.get("iou", 0.0))
    lr_val = metrics.get("lr", 0.0)

    sep_bullet = " · " if UNICODE_ENABLED else " - "
    train_stats = (
        f"loss {Ansi.paint(f'{loss_val:.2f}', Ansi.WHITE)}{sep_bullet}"
        f"cls {Ansi.paint(f'{cls_val:.2f}', Ansi.WHITE)}{sep_bullet}"
        f"{regression_name} {Ansi.paint(f'{reg_val:.2f}', Ansi.WHITE)}{sep_bullet}"
        f"iou {Ansi.paint(f'{iou_val:.2f}', Ansi.WHITE)}{sep_bullet}"
        f"lr {Ansi.paint(f'{lr_val:.2e}', Ansi.AMBER)}"
    )

    if run_progress:
        progress_note = f"{Ansi.paint(run_progress, Ansi.WHITE)}"
        if run_eta:
            progress_note += f" {Ansi.paint(f'(ETA {run_eta})', Ansi.GRAY)}"
        train_stats += f"{sep_bullet}{progress_note}"

    sep = Ansi.paint(f" {GLYPHS['box_v']} ", Ansi.DARK_GRAY)
    summary_line = f"{epoch_badge} {time_str}{sep}{train_stats}"
    print(summary_line)

    if val_metrics:
        arrow = Ansi.paint("->", Ansi.DARK_GRAY)
        val_label = Ansi.paint(f"Val ({val_mode})", Ansi.MINT)
        map50 = val_metrics.get("mAP50", 0.0)
        map50_95 = val_metrics.get("mAP50_95", 0.0)

        val_stats_str = (
            f"mAP50:95 {Ansi.paint(f'{map50_95:.4f}', Ansi.WHITE, Ansi.BOLD)}{sep_bullet}"
            f"mAP50 {Ansi.paint(f'{map50:.4f}', Ansi.WHITE)}"
        )
        best_badge = f" {Ansi.paint('[best]', Ansi.GREEN, Ansi.BOLD)}" if is_best else ""
        val_row = f"  {arrow} {val_label}: {val_stats_str}{best_badge}"
        print(val_row)


def print_completion_card(result: Dict[str, Any]):
    """Print an AGY-styled completion card when training finishes (zero emojis)."""
    term_width = get_terminal_width(80)
    target_width = min(term_width - 1, 80)
    h_bar = GLYPHS["box_h"] * (target_width - 2)

    title = " [OK] Training Complete "
    title_styled = Ansi.paint(title, Ansi.BOLD, Ansi.GREEN)
    title_len = visible_len(title)
    top_bar_right = GLYPHS["box_h"] * max(target_width - 2 - title_len, 2)
    top_line = f"{Ansi.paint(GLYPHS['box_tl'], Ansi.DARK_GRAY)}{title_styled}{Ansi.paint(top_bar_right + GLYPHS['box_tr'], Ansi.DARK_GRAY)}"

    wall_time = format_duration(result.get("training_wall_time_sec", 0.0))
    epochs_done = f"{result.get('epochs_completed', 0)}/{result.get('epochs_requested', 0)}"

    row1_left = f"Duration:   {Ansi.paint(wall_time, Ansi.WHITE)}"
    row1_right = f"Epochs: {Ansi.paint(epochs_done, Ansi.WHITE)}"

    best_score = result.get("best_score")
    best_metric = result.get("best_metric", "mAP50_95")
    score_str = f"{best_score:.4f}" if best_score is not None else "n/a"

    row2_left = f"Best Score: {Ansi.paint(score_str, Ansi.GREEN, Ansi.BOLD)} ({best_metric})"
    row2_right = f"Run:    {Ansi.paint(str(result.get('training_run_id', ''))[:12], Ansi.GRAY)}"

    best_ckpt = result.get("best_checkpoint")
    row3 = f"Best Checkpoint: {Ansi.paint(str(best_ckpt), Ansi.CYAN)}" if best_ckpt else ""

    last_ckpt = result.get("last_checkpoint")
    row4 = f"Last Checkpoint: {Ansi.paint(str(last_ckpt), Ansi.GRAY)}" if last_ckpt else ""

    bot_line = f"{Ansi.paint(GLYPHS['box_bl'] + h_bar + GLYPHS['box_br'], Ansi.DARK_GRAY)}"

    def make_dual_row(left: str, right: str, width: int) -> str:
        inner_space = width - 4
        left_len = visible_len(left)
        right_len = visible_len(right)
        gap = max(inner_space - left_len - right_len, 2)
        inner = f" {left}{' ' * gap}{right} "
        actual_len = visible_len(inner)
        pad_end = max(width - 2 - actual_len, 0)
        return f"{Ansi.paint(GLYPHS['box_v'], Ansi.DARK_GRAY)}{inner}{' ' * pad_end}{Ansi.paint(GLYPHS['box_v'], Ansi.DARK_GRAY)}"

    def make_single_row(content: str, width: int) -> str:
        inner = f" {content} "
        actual_len = visible_len(inner)
        pad_end = max(width - 2 - actual_len, 0)
        return f"{Ansi.paint(GLYPHS['box_v'], Ansi.DARK_GRAY)}{inner}{' ' * pad_end}{Ansi.paint(GLYPHS['box_v'], Ansi.DARK_GRAY)}"

    print()
    print(top_line)
    print(make_dual_row(row1_left, row1_right, target_width))
    print(make_dual_row(row2_left, row2_right, target_width))
    if row3:
        print(make_single_row(row3, target_width))
    if row4:
        print(make_single_row(row4, target_width))
    print(bot_line)
    print()


def log_event(message: str, level: str = "info"):
    """Format and print an event line with AGY glyph and style (strictly no emojis)."""
    if level == "info":
        glyph = Ansi.paint("[INFO]", Ansi.CYAN)
    elif level == "warn":
        glyph = Ansi.paint("[WARN]", Ansi.AMBER)
    elif level == "error":
        glyph = Ansi.paint("[ERROR]", Ansi.RED)
    elif level == "success":
        glyph = Ansi.paint("[OK]", Ansi.GREEN)
    elif level == "step":
        glyph = Ansi.paint("[STEP]", Ansi.PURPLE)
    elif level == "lr":
        glyph = Ansi.paint("[LR]", Ansi.BLUE)
    elif level == "val":
        glyph = Ansi.paint("[VAL]", Ansi.MINT)
    else:
        glyph = Ansi.paint("[*]", Ansi.GRAY)

    print(f"{glyph} {message}")


__all__ = [
    "AgyProgressBar",
    "Ansi",
    "COLOR_ENABLED",
    "GLYPHS",
    "UNICODE_ENABLED",
    "fit_line",
    "format_duration",
    "format_eta",
    "get_terminal_width",
    "log_event",
    "print_completion_card",
    "print_epoch_summary",
    "print_train_card",
    "visible_len",
]
