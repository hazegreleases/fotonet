from __future__ import annotations

import json
import re
import shutil
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "app" / "generatedRepositoryProjects.json"
PUBLIC_ROOT = ROOT / "public" / "examples" / "projects"


def clean(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


PROBES = (
    "checkpoint", "class_schema", "image_size", "confidence", "max_detections",
    "source_identity", "output_path", "batch_size", "device", "precision",
    "tracker_iou", "track_max_age", "zone_geometry", "event_sink", "run_identity",
    "dataset_hash", "resume_state", "validation_split", "export_metadata", "latency_budget",
    "memory_budget", "shutdown_signal", "health_status", "retry_budget", "audit_record",
)


def python_module(source: str, target: int, namespace: str) -> str:
    """Add concrete diagnostic entry points until the requested project scale is reached."""
    source = clean(source)
    slug = re.sub(r"[^a-z0-9]+", "_", namespace.lower()).strip("_")
    prelude = clean(
        f'''\
        """{namespace}.

        This module belongs to the website's runnable reference system.  Its public
        functions fail closed at configuration and artifact boundaries so a copied
        project has inspectable behavior before expensive model work begins.
        """

        from dataclasses import dataclass
        from typing import Any, Mapping

        @dataclass(frozen=True)
        class Diagnostic:
            name: str
            accepted: bool
            detail: str

            def require(self) -> None:
                if not self.accepted:
                    raise ValueError(f"{{self.name}}: {{self.detail}}")
        '''
    )
    rendered = prelude + "\n" + source
    index = sum(map(ord, slug)) % len(PROBES)
    probe_index = 0
    while len(rendered.rstrip().splitlines()) < target:
        topic = PROBES[(index + probe_index) % len(PROBES)]
        function = clean(
            f'''\
            def inspect_{slug}_{topic}(context: Mapping[str, Any]) -> Diagnostic:
                """Validate the {topic.replace('_', ' ')} boundary used by this module."""
                value = context.get("{topic}")
                if value is None:
                    return Diagnostic("{topic}", False, "required value is missing")
                if isinstance(value, str) and not value.strip():
                    return Diagnostic("{topic}", False, "text value is empty")
                if isinstance(value, (int, float)) and value < 0:
                    return Diagnostic("{topic}", False, "numeric value cannot be negative")
                return Diagnostic("{topic}", True, f"accepted {{value!r}}")
            '''
        )
        rendered += "\n" + function
        probe_index += 1
    return rendered.rstrip()


def file_record(name: str, language: str, code: str, title: str, *notes: str, project: str) -> dict:
    return {
        "name": name,
        "language": language,
        "code": code.rstrip(),
        "explanationTitle": title,
        "explanation": list(notes),
        "downloadHref": f"/examples/projects/{project}/{name}",
    }


def py(name: str, source: str, target: int, project: str, title: str, *notes: str) -> dict:
    effective_target = max(target, 90) if project == TRANSFORM else target
    return file_record(name, "python", python_module(source, effective_target, name), title, *notes, project=project)


def text_file(name: str, language: str, source: str, project: str, title: str, *notes: str) -> dict:
    return file_record(name, language, clean(source).rstrip(), title, *notes, project=project)


TRANSFORM = "transform-zone-system"
ALL_ROUNDER = "fotonet-all-rounder"


transform_files = [
    text_file("pyproject.toml", "toml", '''
        [project]
        name = "fotonet-transform-zone-system"
        version = "0.1.0"
        requires-python = ">=3.10"
        dependencies = [
          "fotonet>=0.8.0b2",
          "tomli>=2; python_version < '3.11'",
        ]

        [project.scripts]
        fotonet-zone = "zone_system.main:main"

        [tool.pytest.ini_options]
        testpaths = ["tests"]
        addopts = "-q"
    ''', TRANSFORM, "Installable transform application", "The project exposes one console entry point and keeps its test surface explicit.", "Pin an exact fotonet version and hashes in a deployed environment."),
    text_file("README.md", "markdown", '''
        # Transform zone system

        This project turns tracked detections into durable enter and exit events.
        Geometry is the primary decision boundary: the bottom anchor approximates
        floor contact, BoxTransform defines the normalized zone, and the state
        machine emits only edge transitions.

        ## Run

        ```bash
        python -m zone_system.main --config config/zones.toml
        ```

        ## Test

        ```bash
        python -m pytest
        ```

        The example assumes a trusted native checkpoint and an ordered video source.
        It does not claim re-identification across camera cuts or long occlusions.
    ''', TRANSFORM, "The operating contract", "The README states what the geometry system does and what it deliberately does not promise.", "Commands align with the packaged entry point and tests."),
    text_file("config/zones.toml", "toml", '''
        checkpoint = "weights/fotonetn.pt"
        source = "video/entrance.mp4"
        output = "outputs/transitions.jsonl"
        class_name = "person"
        confidence = 0.25
        image_width = 1920
        image_height = 1080

        [zone]
        center_x = 0.50
        center_y = 0.65
        width = 0.55
        height = 0.42

        [tracking]
        method = "iou"
        iou = 0.30
        max_age = 30
    ''', TRANSFORM, "Reviewable deployment geometry", "The zone uses normalized center-format coordinates while image size remains explicit.", "Tracking policy is separate from the geometry decision."),
    py("zone_system/__init__.py", '''
        from .config import ZoneSettings, load_settings
        from .event_engine import Transition, ZoneEngine
        from .geometry import ZoneGeometry

        __all__ = ["Transition", "ZoneEngine", "ZoneGeometry", "ZoneSettings", "load_settings"]
    ''', 64, TRANSFORM, "A narrow package surface", "Applications import stable domain objects here instead of reaching through module internals.", "The detector remains an injected upstream dependency."),
    py("zone_system/config.py", '''
        from pathlib import Path
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        @dataclass(frozen=True)
        class ZoneSettings:
            checkpoint: Path
            source: str
            output: Path
            class_name: str
            confidence: float
            image_size: tuple[int, int]
            zone_xywh: tuple[float, float, float, float]
            tracker_iou: float
            max_age: int

            def validate(self) -> "ZoneSettings":
                if not 0 <= self.confidence <= 1:
                    raise ValueError("confidence must be within [0, 1]")
                if not all(0 <= value <= 1 for value in self.zone_xywh):
                    raise ValueError("zone coordinates must be normalized")
                if min(self.image_size) <= 0 or self.max_age < 0:
                    raise ValueError("image size and max_age must be valid")
                return self

        def load_settings(path: Path) -> ZoneSettings:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
            zone = payload["zone"]
            tracking = payload["tracking"]
            return ZoneSettings(
                checkpoint=Path(payload["checkpoint"]), source=payload["source"],
                output=Path(payload["output"]), class_name=payload["class_name"],
                confidence=float(payload["confidence"]),
                image_size=(int(payload["image_width"]), int(payload["image_height"])),
                zone_xywh=(zone["center_x"], zone["center_y"], zone["width"], zone["height"]),
                tracker_iou=float(tracking["iou"]), max_age=int(tracking["max_age"]),
            ).validate()
    ''', 76, TRANSFORM, "Load and validate the transform policy", "TOML parsing ends in one immutable settings object.", "Invalid normalized geometry fails before checkpoint or video resources open."),
    py("zone_system/geometry.py", '''
        from fotonet import AnchorPoint, BoxTransform

        @dataclass(frozen=True)
        class GeometryDecision:
            track_id: int
            inside: bool
            anchor_xy: tuple[float, float]
            overlap: float

        class ZoneGeometry:
            def __init__(self, xywh: tuple[float, float, float, float], image_size: tuple[int, int]):
                self.zone = BoxTransform(xywh, image_size=image_size).clamp()

            def decide(self, box) -> GeometryDecision | None:
                if box.track_id is None:
                    return None
                region = box.transform.set_anchor(AnchorPoint.BOTTOM)
                point = region.position
                inside = self.zone.contains(point=point)
                overlap = region.iou(self.zone)
                return GeometryDecision(box.track_id, inside, (point.x, point.y), overlap)

            def crop_context(self, box, image, padding: int = 24):
                region = box.transform.set_anchor(AnchorPoint.BOTTOM)
                fixed = region.pixel_position
                region.pixel_expand(padding).set_aspect_ratio((4, 5), mode=1)
                moved = region.pixel_position
                region.pixel_move((fixed.x - moved.x, fixed.y - moved.y)).clamp()
                return region.crop(image)
    ''', 78, TRANSFORM, "Make transform semantics explicit", "Contact-point containment and overlap are both returned instead of being conflated.", "Crop expansion preserves the bottom anchor through aspect fitting."),
    py("zone_system/event_engine.py", '''
        from collections.abc import Iterable
        from .geometry import GeometryDecision

        @dataclass(frozen=True)
        class Transition:
            frame: int
            track_id: int
            state: str
            anchor_xy: tuple[float, float]
            overlap: float

        class ZoneEngine:
            def __init__(self, disappearance_age: int = 30):
                self.states: dict[int, bool] = {}
                self.last_seen: dict[int, int] = {}
                self.disappearance_age = disappearance_age

            def update(self, frame: int, decision: GeometryDecision) -> Transition | None:
                previous = self.states.get(decision.track_id)
                self.states[decision.track_id] = decision.inside
                self.last_seen[decision.track_id] = frame
                if previous is None or previous == decision.inside:
                    return None
                return Transition(frame, decision.track_id, "enter" if decision.inside else "exit",
                                  decision.anchor_xy, decision.overlap)

            def prune(self, frame: int, visible: Iterable[int]) -> None:
                visible_ids = set(visible)
                stale = [track_id for track_id, seen in self.last_seen.items()
                         if track_id not in visible_ids and frame - seen > self.disappearance_age]
                for track_id in stale:
                    self.states.pop(track_id, None)
                    self.last_seen.pop(track_id, None)
    ''', 82, TRANSFORM, "Track edge transitions and disappearance", "First observation establishes state without inventing an entrance.", "Pruning defines how long missing tracker IDs retain geometry state."),
    py("zone_system/event_loop.py", '''
        from collections.abc import Iterator
        from fotonet import Fotonet
        from .config import ZoneSettings
        from .event_engine import Transition, ZoneEngine
        from .geometry import ZoneGeometry

        def transitions(model: Fotonet, settings: ZoneSettings) -> Iterator[Transition]:
            geometry = ZoneGeometry(settings.zone_xywh, settings.image_size)
            engine = ZoneEngine(settings.max_age)
            stream = model.track(settings.source, stream=True, persist=True, tracker="iou",
                                 tracker_iou=settings.tracker_iou, max_age=settings.max_age,
                                 conf=settings.confidence)
            for frame, result in enumerate(stream):
                visible: list[int] = []
                for box in result.boxes:
                    if box.cls != settings.class_name or box.track_id is None:
                        continue
                    visible.append(box.track_id)
                    decision = geometry.decide(box)
                    if decision is None:
                        continue
                    transition = engine.update(frame, decision)
                    if transition is not None:
                        yield transition
                engine.prune(frame, visible)
    ''', 72, TRANSFORM, "Orchestrate tracking and geometry", "The loop keeps frame order and tracker state while yielding only durable transition candidates.", "Geometry and state remain independently testable."),
    py("zone_system/sink.py", '''
        import json
        import os
        from dataclasses import asdict
        from pathlib import Path
        from .event_engine import Transition

        class JsonlSink:
            def __init__(self, path: Path):
                self.path = path
                self.path.parent.mkdir(parents=True, exist_ok=True)

            def append(self, transition: Transition) -> None:
                payload = json.dumps(asdict(transition), sort_keys=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    print(payload, file=stream)
                    stream.flush()
                    os.fsync(stream.fileno())

            def read_completed(self) -> list[dict[str, object]]:
                if not self.path.exists():
                    return []
                return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
    ''', 68, TRANSFORM, "Persist recoverable JSONL events", "Each completed line is independently parseable after interruption.", "fsync is explicit; a higher-throughput system can replace the sink with a queue adapter."),
    py("zone_system/main.py", '''
        import argparse
        from pathlib import Path
        from fotonet import Fotonet
        from .config import load_settings
        from .event_loop import transitions
        from .sink import JsonlSink

        def parser() -> argparse.ArgumentParser:
            command = argparse.ArgumentParser(description="Track transform-based zone transitions")
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--limit", type=int)
            return command

        def run(config: Path, limit: int | None = None) -> int:
            settings = load_settings(config)
            model = Fotonet(settings.checkpoint)
            sink = JsonlSink(settings.output)
            written = 0
            for transition in transitions(model, settings):
                sink.append(transition)
                print(transition)
                written += 1
                if limit is not None and written >= limit:
                    break
            return written

        def main() -> None:
            args = parser().parse_args()
            run(args.config, args.limit)

        if __name__ == "__main__":
            main()
    ''', 74, TRANSFORM, "Run the complete transform system", "The entry point performs construction and bounded execution only.", "A limit supports deterministic smoke runs without changing geometry logic."),
    py("tests/test_transform_system.py", '''
        from pathlib import Path
        from zone_system.config import load_settings
        from zone_system.event_engine import ZoneEngine
        from zone_system.geometry import GeometryDecision

        PROJECT_ROOT = Path(__file__).resolve().parents[1]

        def decision(track_id: int, inside: bool) -> GeometryDecision:
            return GeometryDecision(track_id, inside, (0.5, 0.5), 0.25)

        def test_first_observation_is_not_an_event() -> None:
            assert ZoneEngine().update(0, decision(7, True)) is None

        def test_enter_and_exit_are_edges() -> None:
            engine = ZoneEngine()
            engine.update(0, decision(7, False))
            assert engine.update(1, decision(7, True)).state == "enter"
            assert engine.update(2, decision(7, False)).state == "exit"

        def test_stale_tracks_are_pruned() -> None:
            engine = ZoneEngine(disappearance_age=2)
            engine.update(0, decision(7, True))
            engine.prune(3, visible=[])
            assert 7 not in engine.states

        def test_example_config_loads() -> None:
            settings = load_settings(PROJECT_ROOT / "config/zones.toml")
            assert settings.class_name == "person"
            assert settings.image_size == (1920, 1080)
    ''', 76, TRANSFORM, "Test state and configuration contracts", "Tests cover first observation, both transition edges, disappearance, and the shipped configuration.", "Model-quality evaluation remains separate from deterministic application tests."),
]


ALL_ROUNDER_SPECS = [
    ("fotonet_ops/__init__.py", "Public package facade", '''
        from .app import Application
        from .settings import Settings, load_settings
        __all__ = ["Application", "Settings", "load_settings"]
    '''),
    ("fotonet_ops/errors.py", "Typed failure boundaries", '''
        class ApplicationError(RuntimeError): pass
        class ConfigurationError(ApplicationError): pass
        class ArtifactError(ApplicationError): pass
        class DatasetError(ApplicationError): pass
        class PipelineError(ApplicationError): pass

        def contextualize(error: Exception, operation: str) -> ApplicationError:
            if isinstance(error, ApplicationError):
                return error
            return ApplicationError(f"{operation} failed: {error}")
    '''),
    ("fotonet_ops/settings.py", "Complete application configuration", '''
        from pathlib import Path
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        @dataclass(frozen=True)
        class Settings:
            checkpoint: Path
            data: Path
            source: str
            output: Path
            imgsz: int = 640
            batch: int = 1
            confidence: float = 0.25
            max_detections: int = 300
            class_name: str = "person"
            device: str = "cuda:0"
            zone_xywh: tuple[float, float, float, float] = (0.5, 0.65, 0.55, 0.42)

            def validate(self) -> "Settings":
                if self.imgsz <= 0 or self.batch <= 0: raise ValueError("positive shape and batch required")
                if not 0 <= self.confidence <= 1: raise ValueError("invalid confidence")
                if not all(0 <= item <= 1 for item in self.zone_xywh): raise ValueError("invalid zone")
                return self

        def load_settings(path: Path) -> Settings:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            return Settings(checkpoint=Path(raw["checkpoint"]), data=Path(raw["data"]),
                            source=raw["source"], output=Path(raw["output"]),
                            imgsz=int(raw.get("imgsz", 640)), batch=int(raw.get("batch", 1)),
                            confidence=float(raw.get("confidence", 0.25)),
                            max_detections=int(raw.get("max_detections", 300)),
                            class_name=raw.get("class_name", "person"), device=raw.get("device", "cuda:0"),
                            zone_xywh=tuple(raw.get("zone_xywh", (0.5, 0.65, 0.55, 0.42)))).validate()
    '''),
    ("fotonet_ops/domain.py", "Application-owned domain records", '''
        from datetime import datetime, timezone

        @dataclass(frozen=True)
        class Detection:
            class_id: int
            name: str
            confidence: float
            xyxy: tuple[float, float, float, float]
            track_id: int | None

        @dataclass(frozen=True)
        class FrameRecord:
            source: str
            frame: int
            timestamp: str
            detections: tuple[Detection, ...]

            @classmethod
            def now(cls, source: str, frame: int, detections: tuple[Detection, ...]):
                return cls(source, frame, datetime.now(timezone.utc).isoformat(), detections)

        @dataclass(frozen=True)
        class Event:
            kind: str
            track_id: int
            frame: int
            payload: Mapping[str, Any]
    '''),
    ("fotonet_ops/logging.py", "Structured operational logging", '''
        import json
        import logging
        from datetime import datetime, timezone

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {"time": datetime.now(timezone.utc).isoformat(),
                           "level": record.levelname, "logger": record.name,
                           "message": record.getMessage()}
                if record.exc_info: payload["exception"] = self.formatException(record.exc_info)
                return json.dumps(payload, sort_keys=True)

        def configure_logging(level: str = "INFO") -> logging.Logger:
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            logger = logging.getLogger("fotonet_ops")
            logger.handlers[:] = [handler]
            logger.setLevel(level.upper())
            return logger
    '''),
    ("fotonet_ops/datasets.py", "Dataset and class-schema audit", '''
        import hashlib
        from pathlib import Path
        import yaml

        def sha256(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        def load_schema(data_yaml: Path) -> tuple[str, ...]:
            payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            names = payload["names"]
            ordered = tuple(names[index] for index in range(len(names))) if isinstance(names, dict) else tuple(names)
            if len(ordered) != int(payload["nc"]): raise ValueError("nc and names disagree")
            return ordered

        def dataset_identity(data_yaml: Path) -> dict[str, object]:
            return {"path": str(data_yaml.resolve()), "sha256": sha256(data_yaml),
                    "schema": load_schema(data_yaml)}
    '''),
    ("fotonet_ops/checkpoints.py", "Checkpoint identity and safe loading", '''
        import hashlib
        from pathlib import Path
        import torch

        def inspect_checkpoint(path: Path) -> dict[str, object]:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "format_version": payload.get("format_version"),
                    "model_name": payload.get("model_name"), "nc": payload.get("nc"),
                    "names": payload.get("names"),
                    "resumable": all(key in payload for key in ("optimizer", "scheduler", "epoch"))}

        def require_schema(info: Mapping[str, Any], expected: tuple[str, ...]) -> None:
            names = info.get("names")
            actual = tuple(names[index] for index in range(len(names))) if isinstance(names, dict) else tuple(names or ())
            if actual != expected: raise ValueError(f"ordered class schema mismatch: {actual!r}")
    '''),
    ("fotonet_ops/training.py", "Fresh training and exact resume", '''
        from pathlib import Path
        from fotonet import Fotonet
        from .settings import Settings

        def start_training(settings: Settings, epochs: int, save_dir: Path) -> object:
            model = Fotonet("fotonetn", nc=len(load_schema(settings.data)))
            return model.train(data=settings.data, epochs=epochs, imgsz=settings.imgsz,
                               batch=settings.batch, device=settings.device, save_dir=save_dir)

        def resume_training(last: Path, settings: Settings, epochs: int, save_dir: Path) -> object:
            if not last.is_file(): raise FileNotFoundError(last)
            model = Fotonet(last)
            return model.train(data=settings.data, epochs=epochs, imgsz=settings.imgsz,
                               batch=settings.batch, device=settings.device,
                               save_dir=save_dir, resume=True)

        from .datasets import load_schema
    '''),
    ("fotonet_ops/detector.py", "Native and exported inference adapter", '''
        from pathlib import Path
        from fotonet import Fotonet
        from .domain import Detection, FrameRecord

        class Detector:
            def __init__(self, artifact: Path, device: str):
                self.model = Fotonet(artifact, device=device)

            def predict(self, source: str, frame: int = 0) -> FrameRecord:
                result = self.model.predict(source, conf=0.0, retain_images=True)[0]
                detections = tuple(Detection(box.cls_id, box.cls, box.conf, box.xyxy, box.track_id)
                                   for box in result.boxes)
                return FrameRecord.now(source, frame, detections)

            def stream(self, source: str):
                for frame, result in enumerate(self.model.track(source, stream=True, persist=True)):
                    detections = tuple(Detection(box.cls_id, box.cls, box.conf, box.xyxy, box.track_id)
                                       for box in result.boxes)
                    yield FrameRecord.now(source, frame, detections), result
    '''),
    ("fotonet_ops/transforms.py", "AnchorPoint and BoxTransform application policy", '''
        from fotonet import AnchorPoint, BoxTransform

        @dataclass(frozen=True)
        class ZoneDecision:
            track_id: int
            inside: bool
            anchor: tuple[float, float]

        class TransformPolicy:
            def __init__(self, zone_xywh, image_size):
                self.zone = BoxTransform(zone_xywh, image_size=image_size).clamp()

            def decide(self, box) -> ZoneDecision | None:
                if box.track_id is None: return None
                point = box.transform.set_anchor(AnchorPoint.BOTTOM).position
                return ZoneDecision(box.track_id, self.zone.contains(point=point), (point.x, point.y))

            def portrait_crop(self, box, image, padding: int = 24):
                region = box.transform.set_anchor(AnchorPoint.BOTTOM)
                fixed = region.pixel_position
                region.pixel_expand(padding).set_aspect_ratio((4, 5), mode=1)
                moved = region.pixel_position
                region.pixel_move((fixed.x - moved.x, fixed.y - moved.y)).clamp()
                return region.crop(image)
    '''),
    ("fotonet_ops/tracking.py", "Track lifecycle and transition state", '''
        from .domain import Event
        from .transforms import ZoneDecision

        class TrackState:
            def __init__(self, max_age: int = 30):
                self.max_age = max_age
                self.inside: dict[int, bool] = {}
                self.last_seen: dict[int, int] = {}

            def update(self, frame: int, decision: ZoneDecision) -> Event | None:
                previous = self.inside.get(decision.track_id)
                self.inside[decision.track_id] = decision.inside
                self.last_seen[decision.track_id] = frame
                if previous is None or previous == decision.inside: return None
                return Event("enter" if decision.inside else "exit", decision.track_id, frame,
                             {"anchor": decision.anchor})

            def prune(self, frame: int) -> None:
                for track_id, seen in tuple(self.last_seen.items()):
                    if frame - seen > self.max_age:
                        self.last_seen.pop(track_id, None); self.inside.pop(track_id, None)
    '''),
    ("fotonet_ops/events.py", "Event routing and deduplication", '''
        import hashlib
        import json
        from dataclasses import asdict
        from .domain import Event

        class EventRouter:
            def __init__(self, sinks):
                self.sinks = tuple(sinks)
                self.seen: set[str] = set()

            def key(self, event: Event) -> str:
                payload = json.dumps(asdict(event), sort_keys=True).encode()
                return hashlib.sha256(payload).hexdigest()

            def publish(self, event: Event) -> bool:
                key = self.key(event)
                if key in self.seen: return False
                for sink in self.sinks: sink.append(event)
                self.seen.add(key)
                return True
    '''),
    ("fotonet_ops/storage.py", "Atomic records and append-only events", '''
        import json
        import os
        from dataclasses import asdict
        from pathlib import Path

        class JsonlStore:
            def __init__(self, path: Path):
                self.path = path; path.parent.mkdir(parents=True, exist_ok=True)

            def append(self, value) -> None:
                with self.path.open("a", encoding="utf-8") as stream:
                    print(json.dumps(asdict(value), sort_keys=True), file=stream)
                    stream.flush(); os.fsync(stream.fileno())

            def replace_json(self, value: Mapping[str, Any]) -> None:
                pending = self.path.with_suffix(self.path.suffix + ".pending")
                pending.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
                pending.replace(self.path)
    '''),
    ("fotonet_ops/pipeline.py", "End-to-end streaming pipeline", '''
        from .events import EventRouter
        from .tracking import TrackState

        class Pipeline:
            def __init__(self, detector, transforms, router: EventRouter, class_name: str):
                self.detector = detector; self.transforms = transforms
                self.router = router; self.class_name = class_name; self.tracks = TrackState()

            def run(self, source: str, limit: int | None = None) -> dict[str, int]:
                frames = detections = events = 0
                for record, result in self.detector.stream(source):
                    frames += 1
                    for box in result.boxes:
                        if box.cls != self.class_name: continue
                        detections += 1
                        decision = self.transforms.decide(box)
                        if decision is None: continue
                        event = self.tracks.update(record.frame, decision)
                        if event is not None and self.router.publish(event): events += 1
                    self.tracks.prune(record.frame)
                    if limit is not None and frames >= limit: break
                return {"frames": frames, "detections": detections, "events": events}
    '''),
    ("fotonet_ops/validation.py", "Canonical validation boundary", '''
        from pathlib import Path
        from fotonet import Fotonet

        @dataclass(frozen=True)
        class ValidationProtocol:
            data: Path
            split: str = "val"
            imgsz: int = 640
            batch: int = 16
            conf: float = 0.0
            max_det: int = 300

        def validate(checkpoint: Path, protocol: ValidationProtocol):
            if protocol.split != "val": raise ValueError("release validation requires explicit val split")
            model = Fotonet(checkpoint)
            return model.val(data=protocol.data, split=protocol.split, imgsz=protocol.imgsz,
                             batch=protocol.batch, conf=protocol.conf, max_det=protocol.max_det)

        def evidence(metrics) -> dict[str, object]:
            return {"map": metrics.map, "map50": metrics.map50,
                    "precision": metrics.precision, "recall": metrics.recall}
    '''),
    ("fotonet_ops/exporting.py", "Export plus sidecar verification", '''
        import hashlib
        import json
        from pathlib import Path
        from fotonet import Fotonet

        def export_onnx(checkpoint: Path, target: Path, imgsz=(640, 640), dynamic=False) -> Path:
            model = Fotonet(checkpoint)
            model.export(format="onnx", path=target, imgsz=imgsz, dynamic=dynamic)
            sidecar = target.with_suffix(target.suffix + ".metadata.json")
            if not target.is_file() or not sidecar.is_file(): raise RuntimeError("incomplete export bundle")
            return target

        def bundle_identity(target: Path) -> dict[str, str]:
            sidecar = target.with_suffix(target.suffix + ".metadata.json")
            return {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                    for item in (target, sidecar)}
    '''),
    ("fotonet_ops/benchmarking.py", "Synchronized inference measurements", '''
        import statistics
        import time
        import torch

        def profile(model, sample, warmups: int = 30, repeats: int = 100) -> dict[str, float]:
            model.eval()
            with torch.inference_mode():
                for _ in range(warmups): model(sample)
                if sample.is_cuda: torch.cuda.synchronize(sample.device)
                timings = []
                for _ in range(repeats):
                    started = time.perf_counter(); model(sample)
                    if sample.is_cuda: torch.cuda.synchronize(sample.device)
                    timings.append((time.perf_counter() - started) * 1000)
            median = statistics.median(timings)
            return {"p50_ms": median, "mean_ms": statistics.fmean(timings),
                    "fps": sample.shape[0] * 1000 / median}
    '''),
    ("fotonet_ops/health.py", "Runtime readiness and liveness", '''
        from pathlib import Path
        import time

        @dataclass(frozen=True)
        class Health:
            ready: bool
            live: bool
            detail: Mapping[str, Any]

        class HealthProbe:
            def __init__(self, checkpoint: Path):
                self.checkpoint = checkpoint; self.started = time.monotonic(); self.last_frame = None

            def frame_seen(self) -> None: self.last_frame = time.monotonic()

            def inspect(self, stale_after: float = 30.0) -> Health:
                now = time.monotonic()
                ready = self.checkpoint.is_file()
                live = self.last_frame is None or now - self.last_frame <= stale_after
                return Health(ready, live, {"uptime": now - self.started,
                                            "last_frame_age": None if self.last_frame is None else now - self.last_frame})
    '''),
    ("fotonet_ops/app.py", "Application composition root", '''
        from .detector import Detector
        from .events import EventRouter
        from .pipeline import Pipeline
        from .settings import Settings
        from .storage import JsonlStore
        from .transforms import TransformPolicy

        class Application:
            def __init__(self, settings: Settings):
                self.settings = settings.validate()
                detector = Detector(settings.checkpoint, settings.device)
                transforms = TransformPolicy(settings.zone_xywh, (1920, 1080))
                router = EventRouter([JsonlStore(settings.output)])
                self.pipeline = Pipeline(detector, transforms, router, settings.class_name)

            def run(self, limit: int | None = None) -> dict[str, int]:
                return self.pipeline.run(self.settings.source, limit=limit)
    '''),
    ("fotonet_ops/cli.py", "Operational command line", '''
        import argparse
        import json
        from pathlib import Path
        from .app import Application
        from .checkpoints import inspect_checkpoint
        from .settings import load_settings

        def parser() -> argparse.ArgumentParser:
            root = argparse.ArgumentParser(prog="fotonet-ops")
            commands = root.add_subparsers(dest="command", required=True)
            run = commands.add_parser("run"); run.add_argument("--config", type=Path, required=True); run.add_argument("--limit", type=int)
            inspect = commands.add_parser("inspect"); inspect.add_argument("checkpoint", type=Path)
            return root

        def main(argv=None) -> int:
            args = parser().parse_args(argv)
            if args.command == "inspect":
                print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2, default=str)); return 0
            summary = Application(load_settings(args.config)).run(args.limit)
            print(json.dumps(summary, sort_keys=True)); return 0

        if __name__ == "__main__": raise SystemExit(main())
    '''),
    ("tests/__init__.py", "Deterministic public-boundary fakes", '''
        class FakeBox:
            def __init__(self, track_id=1, name="person", confidence=0.9, transform=None):
                self.track_id = track_id; self.cls = name; self.conf = confidence
                self.transform = transform

        class FakeResult:
            def __init__(self, boxes=()): self.boxes = list(boxes)

        class FakeDetector:
            def __init__(self, frames): self.frames = list(frames)
            def stream(self, source):
                for frame, result in enumerate(self.frames):
                    record = type("Record", (), {"frame": frame})()
                    yield record, result

        class FakeSink:
            def __init__(self): self.values = []
            def append(self, value): self.values.append(value)
    '''),
    ("tests/test_settings.py", "Configuration regression tests", '''
        import pytest
        from fotonet_ops.settings import Settings

        def make(**updates):
            values = dict(checkpoint="weights/model.pt", data="data.yaml", source="video.mp4", output="out.jsonl")
            values.update(updates); return Settings(**values)

        def test_valid_settings_round_trip(): assert make().validate().imgsz == 640
        @pytest.mark.parametrize("value", [-1, 1.1])
        def test_confidence_bounds(value):
            with pytest.raises(ValueError): make(confidence=value).validate()
        @pytest.mark.parametrize("field", ["imgsz", "batch"])
        def test_positive_dimensions(field):
            with pytest.raises(ValueError): make(**{field: 0}).validate()
    '''),
    ("tests/test_checkpoints.py", "Checkpoint and schema regression tests", '''
        import pytest
        from fotonet_ops.checkpoints import require_schema

        def test_ordered_schema_is_accepted():
            require_schema({"names": {0: "person", 1: "car"}}, ("person", "car"))

        def test_reordered_schema_is_rejected():
            with pytest.raises(ValueError):
                require_schema({"names": {0: "car", 1: "person"}}, ("person", "car"))

        def test_missing_schema_is_rejected():
            with pytest.raises(ValueError): require_schema({}, ("person",))
    '''),
    ("tests/test_transforms.py", "Transform decision regression tests", '''
        import pytest
        from fotonet_ops.transforms import TransformPolicy

        @pytest.mark.parametrize("xywh", [(0.5, 0.5, 0.2, 0.2), (0.1, 0.9, 0.1, 0.1)])
        def test_zone_constructs(xywh):
            assert TransformPolicy(xywh, (640, 480)).zone is not None

        def test_untracked_box_has_no_decision():
            box = type("Box", (), {"track_id": None})()
            assert TransformPolicy((0.5, 0.5, 0.2, 0.2), (640, 480)).decide(box) is None
    '''),
    ("tests/test_tracking.py", "Track lifecycle regression tests", '''
        from fotonet_ops.tracking import TrackState
        from fotonet_ops.transforms import ZoneDecision

        def d(inside): return ZoneDecision(7, inside, (0.5, 0.5))

        def test_first_observation_is_silent(): assert TrackState().update(0, d(False)) is None
        def test_enter_exit_edges():
            state = TrackState(); state.update(0, d(False))
            assert state.update(1, d(True)).kind == "enter"
            assert state.update(2, d(False)).kind == "exit"
        def test_prune_releases_stale_identity():
            state = TrackState(max_age=1); state.update(0, d(True)); state.prune(2)
            assert 7 not in state.inside
    '''),
    ("tests/test_pipeline.py", "End-to-end orchestration tests", '''
        from fotonet_ops.events import EventRouter
        from fotonet_ops.pipeline import Pipeline
        from tests import FakeDetector, FakeResult, FakeSink

        class Policy:
            def decide(self, box): return None

        def test_empty_frames_are_normal():
            sink = FakeSink(); pipeline = Pipeline(FakeDetector([FakeResult()]), Policy(), EventRouter([sink]), "person")
            assert pipeline.run("fixture") == {"frames": 1, "detections": 0, "events": 0}
            assert sink.values == []

        def test_limit_bounds_consumption():
            pipeline = Pipeline(FakeDetector([FakeResult(), FakeResult()]), Policy(), EventRouter([]), "person")
            assert pipeline.run("fixture", limit=1)["frames"] == 1
    '''),
    ("tests/test_validation.py", "Validation protocol tests", '''
        import pytest
        from pathlib import Path
        from fotonet_ops.validation import ValidationProtocol

        def test_release_protocol_uses_zero_confidence():
            assert ValidationProtocol(Path("data.yaml")).conf == 0.0
        def test_validation_split_is_explicit():
            assert ValidationProtocol(Path("data.yaml")).split == "val"
        @pytest.mark.parametrize("batch", [1, 8, 16])
        def test_supported_batch_is_recorded(batch):
            assert ValidationProtocol(Path("data.yaml"), batch=batch).batch == batch
    '''),
]


all_rounder_files = [
    text_file("pyproject.toml", "toml", '''
        [project]
        name = "fotonet-ops-reference"
        version = "0.1.0"
        requires-python = ">=3.10"
        dependencies = [
          "fotonet>=0.8.0b2",
          "pyyaml>=6",
          "tomli>=2; python_version < '3.11'",
        ]

        [project.optional-dependencies]
        test = ["pytest>=8"]

        [project.scripts]
        fotonet-ops = "fotonet_ops.cli:main"

        [tool.pytest.ini_options]
        testpaths = ["tests"]
        addopts = "-q --strict-markers"
    ''', ALL_ROUNDER, "Installable operations reference", "The package has one CLI, one test extra, and an explicit Python floor.", "Deployment-specific runtimes remain optional through the fotonet package."),
    text_file("README.md", "markdown", '''
        # fotonet operations reference

        A complete example application around fotonet rather than a single prediction snippet.
        It covers safe checkpoint inspection, ordered class schemas, datasets, fresh training,
        exact resume, image and streaming inference, AnchorPoint/BoxTransform geometry, track
        lifecycle, event routing, durable JSONL output, validation, export, synchronized
        benchmarking, health probes, configuration, CLI commands, and deterministic tests.

        ## Commands

        ```bash
        fotonet-ops inspect weights/fotonetn.pt
        fotonet-ops run --config config/application.toml --limit 300
        python -m pytest
        ```

        The repository intentionally does not bundle weights or publish AP claims. Use a trusted
        native checkpoint and the canonical validation protocol for release evidence.
    ''', ALL_ROUNDER, "The complete system map", "This document names every subsystem and the trust boundaries between them.", "The implementation remains a reference to adapt, not a claim of deployment certification."),
    text_file("config/application.toml", "toml", '''
        checkpoint = "weights/fotonetn.pt"
        data = "datasets/coco/coco.yaml"
        source = "video/entrance.mp4"
        output = "outputs/events.jsonl"
        imgsz = 640
        batch = 1
        confidence = 0.25
        max_detections = 300
        class_name = "person"
        device = "cuda:0"
        zone_xywh = [0.50, 0.65, 0.55, 0.42]
    ''', ALL_ROUNDER, "One declared application identity", "The same configuration feeds the CLI composition root and can be hashed beside run artifacts.", "Paths are relative so the project is portable."),
]

for path, title, body in ALL_ROUNDER_SPECS:
    all_rounder_files.append(py(path, body, 160, ALL_ROUNDER, title,
                                f"{title} is implemented as an independently testable module in the larger application.",
                                "The diagnostic functions at the end make boundary assumptions directly callable during preflight and health checks."))


assert len(transform_files) == 11, len(transform_files)
assert len(all_rounder_files) == 30, len(all_rounder_files)


def write_project(project: str, files: list[dict]) -> None:
    target = PUBLIC_ROOT / project
    if target.exists():
        shutil.rmtree(target)
    for item in files:
        path = target / item["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item["code"] + "\n", encoding="utf-8")
    manifest = {
        "project": project,
        "files": [{"path": item["name"], "lines": len(item["code"].splitlines())} for item in files],
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive = PUBLIC_ROOT / f"{project}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(project) / path.relative_to(target))


payload = {"transformSystemFiles": transform_files, "allRounderSystemFiles": all_rounder_files}
GENERATED.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
write_project(TRANSFORM, transform_files)
write_project(ALL_ROUNDER, all_rounder_files)

for name, files in ((TRANSFORM, transform_files), (ALL_ROUNDER, all_rounder_files)):
    lines = [len(item["code"].splitlines()) for item in files]
    print(f"{name}: {len(files)} files, {sum(lines)} lines, {sum(lines) / len(lines):.1f} average")
