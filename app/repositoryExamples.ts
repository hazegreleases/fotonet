import generatedProjects from "./generatedRepositoryProjects.json";

export type RepositoryTopic =
  | "inference"
  | "data"
  | "training"
  | "validation"
  | "export"
  | "examples"
  | "models"
  | "checkpoints"
  | "api"
  | "transforms";

export type ExampleFile = {
  name: string;
  language: string;
  code: string;
  explanationTitle: string;
  explanation: string[];
  downloadHref?: string;
};

export type ExampleRepository = {
  tier: 1 | 2 | 3;
  title: string;
  summary: string;
  files: ExampleFile[];
  downloadHref?: string;
};

const py = (name: string, code: string, explanationTitle: string, ...explanation: string[]): ExampleFile => ({
  name, code, explanationTitle, explanation, language: "python",
});

const config = (name: string, language: string, code: string, explanationTitle: string, ...explanation: string[]): ExampleFile => ({
  name, language, code, explanationTitle, explanation,
});

const transformSystemFiles = generatedProjects.transformSystemFiles as ExampleFile[];
const allRounderSystemFiles = generatedProjects.allRounderSystemFiles as ExampleFile[];

export const repositoryExamples: Record<RepositoryTopic, ExampleRepository[]> = {
  inference: [
    {
      tier: 1,
      title: "Single-image prediction",
      summary: "Load one checkpoint, run one image, and save both pixels and structured detections.",
      files: [
        py("predict.py", `from pathlib import Path
from fotonet import Fotonet

checkpoint = Path("weights/fotonetn.pt")
source = Path("images/street.jpg")
output = Path("outputs/street-detected.jpg")

model = Fotonet(checkpoint)
result = model.predict(source, imgsz=640, conf=0.25)[0]
result.save(output)
print(result.to_json())`, "The smallest complete inference program",
          "Prediction returns a list because the API preserves a one-result-per-input contract. Index zero only after intentionally passing one image.",
          "The source image is retained for path inputs, so save() can render without a second image read."),
        config("settings.json", "json", `{
  "checkpoint": "weights/fotonetn.pt",
  "source": "images/street.jpg",
  "imgsz": 640,
  "confidence": 0.25,
  "max_detections": 300
}`, "Keep deployment choices outside the script",
          "A configuration file makes thresholds and paths reviewable without turning the example into a framework.",
          "Confidence is an application operating point. It is not the zero-confidence setting used for ranked COCO validation."),
      ],
    },
    {
      tier: 2,
      title: "Batched folder to JSONL",
      summary: "Process a directory in batches while preserving source order and writing stream-friendly records.",
      files: [
        py("batch_jsonl.py", `import json
from pathlib import Path
from fotonet import Fotonet

model = Fotonet("weights/fotonetn.pt")
results = model.predict("images/inbox", batch=8, conf=0.25, imgsz=640)

destination = Path("outputs/detections.jsonl")
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("w", encoding="utf-8") as stream:
    for index, result in enumerate(results):
        stream.write(json.dumps({
            "input_index": index,
            "detections": json.loads(result.to_json()),
        }) + "\\n")`, "Batch the model, not the output contract",
          "The result list remains aligned with discovered inputs. Writing one JSON object per line avoids holding a second aggregate representation in memory.",
          "For a production ingest queue, add a stable source identifier from your own manifest instead of relying only on discovery order."),
        py("read_jsonl.py", `import json
from pathlib import Path

path = Path("outputs/detections.jsonl")
with path.open(encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        people = [d for d in record["detections"] if d["name"] == "person"]
        print(record["input_index"], len(people))`, "Consume records incrementally",
          "JSON Lines is useful when a run may be interrupted or tailed by another process. Every completed line is independently parseable.",
          "Filtering by class name is readable, but class IDs are preferable when the ordered class schema is already pinned."),
        config("batch.json", "json", `{
  "checkpoint": "weights/fotonetn.pt",
  "source": "images/inbox",
  "output": "outputs/detections.jsonl",
  "batch": 8,
  "imgsz": 640,
  "confidence": 0.25
}`, "Pin a repeatable batch job",
          "Paths, batch size, image size, and operating threshold become reviewable inputs instead of edits inside the worker.",
          "A real service should validate this document before loading a checkpoint or scanning source files."),
      ],
    },
    {
      tier: 3,
      title: "Bounded video worker",
      summary: "Stream frames, preserve tracker state, and separate inference from downstream event handling.",
      files: [
        py("src/video_worker.py", `from collections.abc import Iterator
from fotonet import Fotonet, Results

def tracked_frames(checkpoint: str, source: str) -> Iterator[Results]:
    model = Fotonet(checkpoint)
    yield from model.track(
        source,
        stream=True,
        persist=True,
        tracker="iou",
        tracker_iou=0.30,
        max_age=30,
        conf=0.25,
        imgsz=640,
    )

for frame_index, result in enumerate(tracked_frames(
    "weights/fotonetn.pt", "video/input.mp4"
)):
    for box in result.boxes:
        print(frame_index, box.track_id, box.cls, box.conf)`, "Keep the video path incremental",
          "stream=True prevents accumulation of every frame result. persist=True carries the built-in IoU tracker state across the ordered stream.",
          "The tracker is same-class rectangle association, not re-identification. Long occlusions and camera cuts require an application policy."),
        py("src/event_sink.py", `import json
from pathlib import Path

class JsonlSink:
    def __init__(self, path: str):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.stream = target.open("a", encoding="utf-8", buffering=1)

    def emit(self, event: dict) -> None:
        self.stream.write(json.dumps(event, separators=(",", ":")) + "\\n")

    def close(self) -> None:
        self.stream.close()`, "Make the event boundary explicit",
          "A line-buffered append-only sink makes completed events durable enough for inspection while the video continues.",
          "Add retry, rotation, and delivery acknowledgements at the integration boundary; those concerns do not belong inside model prediction."),
        py("src/__init__.py", `"""Streaming inference application package."""`, "Mark the application boundary",
          "The package marker makes imports deterministic when the project is launched with python -m.",
          "Keep model integration code under src so entry points, tests, and deployment wrappers all use the same implementation."),
        py("src/settings.py", `from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    checkpoint: str = "weights/fotonetn.pt"
    source: str = "video/input.mp4"
    output: str = "outputs/detections.jsonl"
    confidence: float = 0.25
    tracker_iou: float = 0.30
    max_age: int = 30`, "Keep runtime policy in one immutable object",
          "Paths, thresholds, and association lifetime become explicit inputs rather than scattered literals.",
          "A deployment wrapper can construct Settings from environment variables or a validated configuration file without changing the pipeline."),
        py("main.py", `from src.event_sink import JsonlSink
from src.settings import Settings
from src.video_worker import tracked_frames

def run(settings: Settings) -> None:
    sink = JsonlSink(settings.output)
    try:
        for frame_index, result in enumerate(tracked_frames(
            settings.checkpoint, settings.source
        )):
            for box in result.boxes:
                sink.emit({
                    "frame": frame_index,
                    "track_id": box.track_id,
                    "class": box.cls,
                    "confidence": box.conf,
                    "xyxy": box.xyxy,
                })
    finally:
        sink.close()

if __name__ == "__main__":
    run(Settings())`, "Connect inference, tracking, configuration, and output",
          "This is the system entry point: it owns lifecycle and cleanup while the worker owns model iteration and the sink owns persistence.",
          "Run it with python -m main. Replace JsonlSink behind the same emit boundary when integrating a queue or database."),
      ],
    },
  ],

  data: [
    {
      tier: 1,
      title: "Label sanity check",
      summary: "Validate normalized YOLO rows before a training process touches the dataset.",
      files: [
        py("check_labels.py", `from pathlib import Path

def check_file(path: Path, nc: int) -> list[str]:
    errors = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields")
            continue
        cls, x, y, w, h = map(float, parts)
        if not cls.is_integer() or not 0 <= int(cls) < nc:
            errors.append(f"{path}:{line_number}: class out of range")
        if not all(0.0 <= value <= 1.0 for value in (x, y, w, h)):
            errors.append(f"{path}:{line_number}: coordinates not normalized")
    return errors

for label in Path("datasets/animals/labels").rglob("*.txt"):
    for problem in check_file(label, nc=2):
        print(problem)`, "Fail cheaply before loading pixels",
          "This checks row shape, class bounds, and normalized coordinates. It intentionally does not silently repair annotations.",
          "FOTO-NET still performs its own annotation-policy audit during data loading; this script gives dataset authors faster local feedback."),
        config("data.yaml", "yaml", `path: datasets/animals
train: images/train
val: images/val
nc: 2
names:
  0: cat
  1: dog`, "One ordered class schema",
          "The numeric keys are contiguous and their order is semantic identity. Reordering names is a schema change even when nc stays equal.",
          "Keep validation explicit. The loader will not silently evaluate on the training split."),
      ],
    },
    {
      tier: 2,
      title: "Deterministic split builder",
      summary: "Create repeatable manifests without moving the authoritative image files.",
      files: [
        py("build_split.py", `import random
from pathlib import Path

root = Path("datasets/animals/images/all")
images = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
rng = random.Random(20260811)
rng.shuffle(images)
cut = round(len(images) * 0.8)

out = Path("datasets/animals/splits")
out.mkdir(parents=True, exist_ok=True)
(out / "train.txt").write_text("\\n".join(map(str, images[:cut])) + "\\n")
(out / "val.txt").write_text("\\n".join(map(str, images[cut:])) + "\\n")`, "Split from a sorted source list and fixed seed",
          "Sorting before seeded shuffling removes filesystem enumeration order from the result.",
          "Text manifests avoid duplicate image copies and make the exact membership easy to diff or hash."),
        py("verify_split.py", `from pathlib import Path

def members(path: str) -> set[Path]:
    return {Path(line.strip()).resolve() for line in Path(path).read_text().splitlines() if line.strip()}

train = members("datasets/animals/splits/train.txt")
val = members("datasets/animals/splits/val.txt")
overlap = train & val
missing = {path for path in train | val if not path.is_file()}

if overlap or missing:
    raise SystemExit(f"overlap={len(overlap)} missing={len(missing)}")
print(f"train={len(train)} val={len(val)}")`, "Verify disjointness and existence",
          "A seeded split is still invalid if manifests overlap or point at missing files.",
          "For grouped data such as video frames, split by sequence or subject before generating image-level manifests to prevent leakage."),
      ],
    },
    {
      tier: 3,
      title: "Dataset identity manifest",
      summary: "Hash ordered paths, labels, and class names so a resumed run can detect data drift.",
      files: [
        py("tools/dataset_manifest.py", `import hashlib
import json
from pathlib import Path

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

root = Path("datasets/animals")
records = []
for image in sorted((root / "images").rglob("*.jpg")):
    label = root / "labels" / image.relative_to(root / "images").with_suffix(".txt")
    records.append({
        "image": image.relative_to(root).as_posix(),
        "image_sha256": digest(image),
        "label": label.relative_to(root).as_posix(),
        "label_sha256": digest(label) if label.exists() else None,
    })

payload = {"schema": ["cat", "dog"], "records": records}
Path("dataset-manifest.json").write_text(json.dumps(payload, indent=2) + "\\n")`, "Record content identity, not only counts",
          "Counts cannot detect reordered classes, replaced pixels, or changed labels. Content hashes can.",
          "Large datasets may use a metadata database or sampled integrity policy, but the manifest format should remain deterministic and versioned."),
        py("tools/compare_manifests.py", `import json
from pathlib import Path

before = json.loads(Path("manifest-before.json").read_text())
after = json.loads(Path("dataset-manifest.json").read_text())

if before["schema"] != after["schema"]:
    raise SystemExit("ordered class schema changed")

old = {item["image"]: item for item in before["records"]}
new = {item["image"]: item for item in after["records"]}
print("added", sorted(new.keys() - old.keys()))
print("removed", sorted(old.keys() - new.keys()))
print("changed", sorted(key for key in old.keys() & new.keys() if old[key] != new[key]))`, "Explain drift instead of reporting one opaque hash",
          "The comparison identifies additions, removals, and content changes while treating class order as a separate hard contract.",
          "Resume should remain strict. If data intentionally changes, start a fresh run so sampler and optimizer continuity are not misrepresented."),
        py("tools/__init__.py", `"""Dataset identity and audit tools."""`, "Make the audit tools importable",
          "The package boundary lets CI, local commands, and dataset publishing jobs reuse the same policy functions.",
          "It also prevents one-off notebook logic from becoming the only record of how a dataset was accepted."),
        py("tools/dataset_policy.py", `from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetPolicy:
    class_names: tuple[str, ...] = ("cat", "dog")
    allowed_suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    allow_missing_labels: bool = False
    require_disjoint_splits: bool = True

POLICY = DatasetPolicy()`, "Define acceptance policy once",
          "Class order, supported image types, and missing-label behavior are semantic dataset inputs.",
          "Import this object in audit, manifest, and publishing commands so they cannot drift independently."),
        py("run_audit.py", `import subprocess
import sys
from pathlib import Path
from tools.dataset_policy import POLICY

def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)

if not Path("datasets/animals").is_dir():
    raise SystemExit("dataset root is missing")
if not POLICY.class_names:
    raise SystemExit("ordered class schema is empty")

run("check_labels.py")
run("tools/dataset_manifest.py")
if Path("manifest-before.json").is_file():
    run("tools/compare_manifests.py")
print("dataset audit complete")`, "Turn separate checks into one release gate",
          "The command fails on the first rejected stage and only compares drift when a prior manifest exists.",
          "A production dataset publication job would archive the manifest and audit output beside the immutable dataset version."),
      ],
    },
  ],

  training: [
    {
      tier: 1,
      title: "Minimal fresh run",
      summary: "Construct a canonical model and start one explicit training run.",
      files: [
        py("train_basic.py", `from fotonet import Fotonet

model = Fotonet("fotonetn", nc=2)
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cuda:0",
    save_dir="runs/animals-nano",
)`, "Start from an architecture, not an assumed weight",
          "A canonical model name constructs the production graph. nc must match the ordered dataset schema.",
          "Use a checkpoint path instead when intentionally starting from pretrained weights, and keep resume separate from weights-only initialization."),
        config("data.yaml", "yaml", `path: datasets/animals
train: images/train
val: images/val
nc: 2
names:
  0: cat
  1: dog`, "Training and validation share one schema",
          "The validation source is declared independently and is never inferred from train.",
          "Use repository-relative or deployment-specific paths in published examples; avoid local absolute paths."),
      ],
    },
    {
      tier: 2,
      title: "Configured run and resume",
      summary: "Keep fresh-run options and exact continuation in separate entry points.",
      files: [
        py("start.py", `from fotonet import Fotonet

model = Fotonet("fotonetn", nc=2)
model.train(
    data="data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    optimizer="adamw",
    lr0=0.001,
    val_period=1,
    save_dir="runs/animals-nano",
)`, "Keep the fresh run declarative",
          "The epoch count is the total training horizon. Optimizer and learning-rate choices belong to the run identity recorded beside its checkpoints.",
          "Dry-run the public launcher when you want strict preflight checks before dataset or optimizer construction."),
        py("resume.py", `from fotonet import Fotonet

last = "runs/animals-nano/fotonet_last.pt"
model = Fotonet(last)
model.train(
    data="data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    save_dir="runs/animals-nano",
    resume=True,
)`, "Resume the same run from its full last checkpoint",
          "Resume restores optimizer, scheduler, scaler, EMA, RNG, and progress state. A slim best checkpoint is not an exact continuation source.",
          "Do not silently change class order, graph, dataset membership, or total schedule while claiming continuation."),
        py("check_run.py", `import json
from pathlib import Path

run = Path("runs/animals-nano")
required = [run / "fotonet_last.pt", run / "launch-manifest.json"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"run is not resumable: missing {missing}")
manifest = json.loads((run / "launch-manifest.json").read_text())
print({"model": manifest["model"], "epochs": manifest["epochs"]})`, "Check continuity inputs before resume",
          "The preflight ensures the full last checkpoint and the run's declared launch identity are both present.",
          "The package loader remains authoritative for optimizer, scheduler, class-schema, and graph compatibility."),
      ],
    },
    {
      tier: 3,
      title: "Launch manifest guard",
      summary: "Pin the command, dataset, graph choice, and environment before a long run starts.",
      files: [
        py("tools/write_manifest.py", `import hashlib
import json
import platform
from pathlib import Path
import torch

def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

manifest = {
    "model": "fotonetn",
    "data": "data.yaml",
    "data_sha256": sha256("data.yaml"),
    "epochs": 300,
    "batch": 16,
    "imgsz": 640,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
}
Path("runs/animals-nano/launch.json").write_text(json.dumps(manifest, indent=2) + "\\n")`, "Record inputs before consuming compute",
          "The manifest is evidence about intended inputs, not a replacement for the checkpoint's own resumable state.",
          "Include source revision and package version in a real release run. Never place credentials or machine-private paths in a public manifest."),
        py("tools/assert_resume.py", `import hashlib
import json
from pathlib import Path

run = Path("runs/animals-nano")
manifest = json.loads((run / "launch.json").read_text())
current = hashlib.sha256(Path(manifest["data"]).read_bytes()).hexdigest()

if current != manifest["data_sha256"]:
    raise SystemExit("data.yaml changed; start a fresh run or restore it")
if not (run / "fotonet_last.pt").is_file():
    raise SystemExit("the exact run has no full last checkpoint")
print("resume inputs match the launch manifest")`, "Make accidental drift a hard failure",
          "This lightweight guard catches one common failure mode before the model API performs its deeper checkpoint checks.",
          "The dedicated public train.py already enforces stronger run binding; use this pattern when integrating it into a larger scheduler."),
        py("run_config.py", `from dataclasses import dataclass

@dataclass(frozen=True)
class RunConfig:
    model: str = "fotonetn"
    data: str = "data.yaml"
    run_dir: str = "runs/animals-nano"
    epochs: int = 300
    batch: int = 16
    imgsz: int = 640

CONFIG = RunConfig()`, "Freeze the scheduler-facing run configuration",
          "The same values are used for fresh launch and exact continuation, which removes a common source of resume drift.",
          "Secrets and machine-specific credentials do not belong in this object or its persisted manifest."),
        py("launch.py", `import subprocess
import sys
from pathlib import Path
from run_config import CONFIG

run_dir = Path(CONFIG.run_dir)
resume = (run_dir / "fotonet_last.pt").is_file()
subprocess.run([sys.executable, "tools/write_manifest.py"], check=True)
if resume:
    subprocess.run([sys.executable, "tools/assert_resume.py"], check=True)

command = [
    sys.executable, "train.py",
    "--model", CONFIG.model,
    "--data", CONFIG.data,
    "--run-dir", CONFIG.run_dir,
    "--epochs", str(CONFIG.epochs),
    "--batch", str(CONFIG.batch),
    "--imgsz", str(CONFIG.imgsz),
]
if resume:
    command.append("--resume")
subprocess.run(command, check=True)`, "Use one launcher for fresh and interrupted jobs",
          "The exact run directory decides whether continuation is possible; the command never searches unrelated runs for a latest checkpoint.",
          "The public train.py remains the authoritative strict launcher. This wrapper integrates it into an external scheduler without duplicating training logic."),
        py("monitor.py", `import time
from pathlib import Path
from run_config import CONFIG

checkpoint = Path(CONFIG.run_dir) / "fotonet_last.pt"
last_size = None
while True:
    if checkpoint.is_file():
        size = checkpoint.stat().st_size
        if size != last_size:
            print(f"checkpoint updated: {size:,} bytes", flush=True)
            last_size = size
    time.sleep(30)`, "Observe progress without touching training state",
          "The monitor reads file metadata only and can run beside a scheduler without importing the model or checkpoint.",
          "Real monitoring should also consume structured training logs and alert on stalled progress; it must never rewrite an active checkpoint."),
      ],
    },
  ],

  validation: [
    {
      tier: 1,
      title: "One declared validation run",
      summary: "Evaluate a checkpoint on an explicit split and print ranked metrics.",
      files: [
        py("validate.py", `from fotonet import Fotonet

model = Fotonet("weights/fotonetn.pt")
metrics = model.val(
    data="data.yaml",
    imgsz=640,
    batch=8,
    conf=0.0,
    max_det=100,
    operating_conf=0.25,
    operating_iou=0.50,
)
print("mAP50-95", metrics["mAP50_95"])
print("mAP50", metrics["mAP50"])`, "Separate ranked AP from the operating point",
          "conf=0.0 retains finite candidates for score-ranked evaluation; max_det then applies the declared cap.",
          "Operating precision and recall answer a deployment-threshold question and should not replace ranked AP."),
        config("protocol.json", "json", `{
  "split": "val",
  "imgsz": 640,
  "batch": 8,
  "rank_conf": 0.0,
  "max_det": 100,
  "operating_conf": 0.25,
  "operating_iou": 0.5
}`, "Write the protocol next to the report",
          "A metric without its dataset split, image size, and ranking policy cannot support a reproducible claim.",
          "The evaluator backend and annotation policy should also be retained in release evidence."),
      ],
    },
    {
      tier: 2,
      title: "Checkpoint comparison",
      summary: "Run several weights through one frozen protocol and preserve raw reports.",
      files: [
        py("compare.py", `import json
from pathlib import Path
from fotonet import Fotonet

checkpoints = sorted(Path("weights/candidates").glob("*.pt"))
reports = {}
for checkpoint in checkpoints:
    reports[checkpoint.name] = Fotonet(checkpoint).val(
        data="data.yaml", imgsz=640, batch=8,
        conf=0.0, max_det=100,
        operating_conf=0.25, operating_iou=0.50,
    )

Path("reports").mkdir(exist_ok=True)
Path("reports/comparison.json").write_text(json.dumps(reports, indent=2, default=float) + "\\n")`, "Hold the protocol constant across candidates",
          "Only the checkpoint changes inside the loop. This avoids accidental per-model threshold or image-size tuning.",
          "For official COCO evidence, use the repository's strict release validator rather than treating this convenience comparison as canonical."),
        py("rank.py", `import json
from pathlib import Path

reports = json.loads(Path("reports/comparison.json").read_text())
ranked = sorted(
    reports.items(),
    key=lambda item: item[1]["mAP50_95"],
    reverse=True,
)
for name, report in ranked:
    print(f"{name:32} {report['mAP50_95']:.4f}")`, "Rank one declared primary metric",
          "Choose the selection metric before reading the result. Post-hoc metric choice makes the comparison difficult to interpret.",
          "Retain secondary metrics and per-class behavior for diagnosis, but do not quietly replace the selection rule."),
      ],
    },
    {
      tier: 3,
      title: "Release evidence bundle",
      summary: "Bind a report to its exact checkpoint, dataset declaration, and protocol.",
      files: [
        py("tools/build_evidence.py", `import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

checkpoint = Path("weights/fotonetn.pt")
report = Path("reports/coco-val2017.json")
evidence = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "checkpoint": checkpoint.name,
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    "dataset": "COCO val2017",
    "images": 5000,
    "imgsz": 640,
    "max_det": 100,
}
Path("reports/evidence.json").write_text(json.dumps(evidence, indent=2) + "\\n")`, "Bind claims to immutable artifacts",
          "The checkpoint hash prevents a filename from standing in for identity. The report hash does the same for evaluator output.",
          "Add the exact command, package revision, category mapping, evaluator backend, and annotation audit before publishing an AP claim."),
        py("tools/verify_evidence.py", `import hashlib
import json
from pathlib import Path

evidence = json.loads(Path("reports/evidence.json").read_text())
for key, path in (
    ("checkpoint_sha256", Path("weights") / evidence["checkpoint"]),
    ("report_sha256", Path("reports/coco-val2017.json")),
):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != evidence[key]:
        raise SystemExit(f"hash mismatch: {path}")
print("evidence bundle verified")`, "Verify before publishing or comparing",
          "This check detects replaced artifacts after evaluation.",
          "A valid hash proves file identity, not that the evaluation protocol was correct; both provenance and protocol review remain necessary."),
        py("protocol.py", `from dataclasses import dataclass

@dataclass(frozen=True)
class ValidationProtocol:
    data: str = "data.yaml"
    imgsz: int = 640
    batch: int = 8
    rank_conf: float = 0.0
    max_det: int = 100
    operating_conf: float = 0.25
    operating_iou: float = 0.50

PROTOCOL = ValidationProtocol()`, "Represent the evaluation protocol as data",
          "The primary ranking threshold and deployment operating point remain separate fields.",
          "The release evidence generator should serialize these exact values rather than restating them manually."),
        py("run_validation.py", `import json
from pathlib import Path
from fotonet import Fotonet
from protocol import PROTOCOL

model = Fotonet("weights/fotonetn.pt")
metrics = model.val(
    data=PROTOCOL.data,
    imgsz=PROTOCOL.imgsz,
    batch=PROTOCOL.batch,
    conf=PROTOCOL.rank_conf,
    max_det=PROTOCOL.max_det,
    operating_conf=PROTOCOL.operating_conf,
    operating_iou=PROTOCOL.operating_iou,
)
Path("reports").mkdir(exist_ok=True)
Path("reports/coco-val2017.json").write_text(json.dumps(metrics, indent=2, default=float))`, "Produce the report from one frozen protocol",
          "The checkpoint, data declaration, ranking settings, and operating point enter through one auditable path.",
          "For an official release, run the repository's strict canonical validator and preserve its complete output."),
        py("release_check.py", `import subprocess
import sys
from pathlib import Path

required = [
    Path("weights/fotonetn.pt"),
    Path("reports/coco-val2017.json"),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing release inputs: {missing}")

subprocess.run([sys.executable, "tools/build_evidence.py"], check=True)
subprocess.run([sys.executable, "tools/verify_evidence.py"], check=True)
print("validation evidence is internally consistent")`, "Gate publication on report and artifact identity",
          "The gate refuses to build evidence until both the checkpoint and evaluator report exist, then verifies the resulting hashes.",
          "This is an integrity layer around evaluation, not a substitute for reviewing dataset provenance and canonical protocol output."),
      ],
    },
  ],

  export: [
    {
      tier: 1,
      title: "Static ONNX export",
      summary: "Export a fixed 640-square graph and inspect the adjacent metadata contract.",
      files: [
        py("export_onnx.py", `from fotonet import Fotonet

model = Fotonet("weights/fotonetn.pt")
artifact = model.export(
    format="onnx",
    path="exports/fotonetn-640.onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
)
print(artifact["artifact"])
print(artifact["metadata"])`, "Export and parity-check through the public API",
          "Static export records a fixed input shape and verifies native versus backend output within the exporter protocol.",
          "Keep the ONNX metadata sidecar beside the graph; it carries class, preprocessing, stride, and output-layout identity."),
        py("inspect_metadata.py", `import json
from pathlib import Path

sidecar = Path("exports/fotonetn-640.onnx.metadata.json")
metadata = json.loads(sidecar.read_text())
required = {"format", "nc", "names", "imgsz", "strides"}
missing = required - metadata.keys()
if missing:
    raise SystemExit(f"missing metadata: {sorted(missing)}")
print(json.dumps(metadata, indent=2))`, "Treat metadata as part of the artifact",
          "A graph file alone does not communicate class names or preprocessing assumptions.",
          "Do not hand-edit the sidecar. Re-export the checkpoint so graph and metadata are generated from one source."),
      ],
    },
    {
      tier: 2,
      title: "Dynamic-shape export",
      summary: "Build one ONNX graph for variable image dimensions and test more than the nominal shape.",
      files: [
        py("export_dynamic.py", `from fotonet import Fotonet

model = Fotonet("weights/fotonetn.pt")
model.export(
    format="onnx",
    path="exports/fotonetn-dynamic.onnx",
    imgsz=(640, 960),
    batch=1,
    dynamic=True,
)`, "Declare variability at export time",
          "A static graph should reject incompatible explicit image sizes. dynamic=True is the intentional contract for variable height and width.",
          "Dynamic compatibility does not imply equal latency at every shape; benchmark the actual deployment envelope."),
        py("smoke_shapes.py", `from fotonet import Fotonet

runtime = Fotonet("exports/fotonetn-dynamic.onnx")
for shape in ((512, 512), (640, 960), (736, 1280)):
    result = runtime.predict("images/sample.jpg", imgsz=shape, conf=0.25)[0]
    print(shape, len(result))`, "Exercise several supported shapes",
          "Loading the exported artifact through Fotonet applies its recorded metadata and runtime validation.",
          "A smoke test proves execution only. Use representative images and numerical parity checks before accepting a deployment backend."),
        config("shapes.json", "json", `{
  "artifact": "exports/fotonetn-dynamic.onnx",
  "shapes": [[512, 512], [640, 960], [736, 1280]],
  "batch": 1,
  "precision": "fp32"
}`, "Declare the deployment envelope",
          "The smoke matrix records exactly which shapes and precision the target application promises to support.",
          "Treat every new execution provider, precision, or batch range as a separate compatibility result."),
      ],
    },
    {
      tier: 3,
      title: "Artifact bundle verification",
      summary: "Package a graph and sidecar with hashes, then verify the bundle before deployment.",
      files: [
        py("tools/bundle.py", `import hashlib
import json
from pathlib import Path

files = [
    Path("exports/fotonetn-640.onnx"),
    Path("exports/fotonetn-640.onnx.metadata.json"),
]
manifest = {
    "schema": 1,
    "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
}
Path("exports/manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n")`, "Hash the graph and its contract together",
          "The manifest makes partial or mismatched artifact copies detectable.",
          "Sign the manifest or publish it through a trusted release channel when supply-chain authenticity matters."),
        py("tools/verify_bundle.py", `import hashlib
import json
from pathlib import Path

root = Path("exports")
manifest = json.loads((root / "manifest.json").read_text())
for name, expected in manifest["files"].items():
    path = root / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"artifact mismatch: {name}")
print("bundle verified")`, "Fail closed on missing or changed files",
          "Verification belongs before runtime construction, not after a prediction has already crossed the trust boundary.",
          "Exported programs and their metadata are trusted build inputs. Do not load artifacts from an unverified source."),
        py("config.py", `from dataclasses import dataclass

@dataclass(frozen=True)
class ExportConfig:
    checkpoint: str = "weights/fotonetn.pt"
    artifact: str = "exports/fotonetn-640.onnx"
    imgsz: int = 640
    batch: int = 1
    dynamic: bool = False

CONFIG = ExportConfig()`, "Pin the graph and shape contract",
          "Static versus dynamic shape is a deployment decision and belongs in the reviewed export configuration.",
          "Backend-specific precision and calibration settings should be added here only when their validation evidence also exists."),
        py("export_model.py", `from fotonet import Fotonet
from config import CONFIG

model = Fotonet(CONFIG.checkpoint)
artifact = model.export(
    format="onnx",
    path=CONFIG.artifact,
    imgsz=CONFIG.imgsz,
    batch=CONFIG.batch,
    dynamic=CONFIG.dynamic,
)
print(artifact)`, "Generate the deployment graph from reviewed settings",
          "The public exporter writes and validates the graph plus its sidecar metadata.",
          "Do not hand-edit either generated file; update configuration and export again."),
        py("deploy_smoke.py", `from pathlib import Path
from fotonet import Fotonet
from config import CONFIG

artifact = Path(CONFIG.artifact)
sidecar = artifact.with_suffix(artifact.suffix + ".metadata.json")
if not artifact.is_file() or not sidecar.is_file():
    raise SystemExit("export bundle is incomplete")

runtime = Fotonet(artifact)
result = runtime.predict("images/smoke.jpg", imgsz=CONFIG.imgsz)[0]
print(f"runtime accepted {len(result)} detections")`, "Load the exact artifact through its deployment contract",
          "The smoke image checks runtime construction, preprocessing metadata, output decoding, and static shape acceptance.",
          "After this passes, build the checksum manifest and run representative numerical and target-hardware tests."),
      ],
    },
  ],

  examples: [
    {
      tier: 1,
      title: "Annotated image command",
      summary: "Wrap one prediction in a small command-line program with explicit paths.",
      files: [
        py("main.py", `import argparse
from pathlib import Path
from fotonet import Fotonet

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("image", type=Path)
parser.add_argument("--output", type=Path, default=Path("outputs/result.jpg"))
args = parser.parse_args()

result = Fotonet(args.checkpoint).predict(args.image, conf=0.25, imgsz=640)[0]
result.save(args.output)
print(result.to_json())`, "A script with a stable invocation",
          "The positional arguments identify immutable inputs; the output defaults to a separate generated directory.",
          "Path validation and richer error messages can be added without changing the inference contract."),
        config("run.txt", "text", `python main.py weights/fotonetn.pt images/example.jpg \
  --output outputs/example-detected.jpg`, "Keep the first run copyable",
          "A complete invocation is more useful than a fragment because it exposes path and output assumptions.",
          "On PowerShell, enter the command on one line or use its native line-continuation syntax."),
      ],
    },
    {
      tier: 2,
      title: "Folder report",
      summary: "Turn a folder into durable JSONL plus an aggregate class count.",
      files: [
        py("folder_report.py", `import json
from collections import Counter
from pathlib import Path
from fotonet import Fotonet

results = Fotonet("weights/fotonetn.pt").predict("images/inbox", batch=8)
counts = Counter()
target = Path("outputs/report.jsonl")
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w", encoding="utf-8") as stream:
    for index, result in enumerate(results):
        detections = json.loads(result.to_json())
        counts.update(item["name"] for item in detections)
        stream.write(json.dumps({"index": index, "detections": detections}) + "\\n")
print(dict(counts))`, "Keep per-image records and aggregate separately",
          "The JSONL remains the inspectable source record; the Counter is only a derived summary.",
          "Attach filenames from an explicit input manifest when index-only identity is insufficient."),
        py("top_classes.py", `import json
from collections import Counter
from pathlib import Path

counts = Counter()
for line in Path("outputs/report.jsonl").read_text().splitlines():
    record = json.loads(line)
    counts.update(item["name"] for item in record["detections"])
for name, count in counts.most_common(10):
    print(f"{name:24} {count}")`, "Derive reports without rerunning inference",
          "Separating expensive prediction from cheap analysis makes iteration faster and audit trails clearer.",
          "Counts describe detections, not unique real-world objects unless tracking and deduplication are part of the application."),
      ],
    },
    {
      tier: 3,
      title: "Complete fotonet operations system",
      summary: "A complete 30-file operations reference spanning the full fotonet lifecycle.",
      downloadHref: "/examples/projects/fotonet-all-rounder.zip",
      files: allRounderSystemFiles.length ? allRounderSystemFiles : [
        py("app/zone_events.py", `from collections.abc import Iterator
from fotonet import AnchorPoint, BoxTransform, Fotonet
from app.policy import ZonePolicy

def watch_zone(
    model: Fotonet,
    source: str,
    zone: BoxTransform,
    policy: ZonePolicy,
) -> Iterator[dict[str, int | str]]:
    states: dict[int, bool] = {}
    for frame, result in enumerate(model.track(source, stream=True, persist=True)):
        visible: set[int] = set()
        for box in result.boxes:
            if box.track_id is None or not policy.accept(box):
                continue
            visible.add(box.track_id)
            point = box.transform.set_anchor(AnchorPoint.BOTTOM).position
            inside = zone.contains(point=point)
            previous = states.get(box.track_id)
            if previous is not None and previous != inside:
                yield {"frame": frame, "track": box.track_id,
                       "state": "enter" if inside else "exit"}
            states[box.track_id] = inside
        states = {key: value for key, value in states.items() if key in visible}`, "Emit transitions, not one event per frame",
          "The bottom anchor approximates floor contact better than the rectangle center for an entrance zone.",
          "State is pruned to visible tracks. A longer disappearance policy needs timestamps or max-age logic consistent with the application."),
        py("app/policy.py", `from dataclasses import dataclass

@dataclass(frozen=True)
class ZonePolicy:
    class_name: str = "person"
    confidence: float = 0.25
    tracker_iou: float = 0.30
    max_age: int = 30

    def accept(self, box) -> bool:
        return box.cls == self.class_name and box.conf >= self.confidence`, "Keep application policy inspectable",
          "A frozen data class prevents scattered magic values and makes the policy easy to serialize beside event logs.",
          "Geometry, association, and business decisions are separate layers; keeping them separate makes false-event diagnosis possible."),
        py("app/config.py", `from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    checkpoint: Path = Path("weights/fotonetn.pt")
    source: str = "entrance.mp4"
    events: Path = Path("outputs/zone-events.jsonl")
    image_size: tuple[int, int] = (1920, 1080)
    zone_xywh: tuple[float, float, float, float] = (0.50, 0.62, 0.55, 0.50)`, "Centralize runtime identity and geometry",
          "The configuration keeps paths, image size, and normalized zone geometry reviewable in one immutable object.",
          "Production deployments can replace this module with validated environment or YAML loading without changing the event engine."),
        py("app/sink.py", `import json
from pathlib import Path
from typing import Any

class JsonlEventSink:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            print(json.dumps(event, sort_keys=True), file=stream)`, "Make event persistence explicit",
          "JSONL is append-friendly and keeps each transition independently recoverable after an interruption.",
          "A database or queue adapter can implement the same append boundary when durability or delivery guarantees increase."),
        py("main.py", `from fotonet import BoxTransform, Fotonet
from app.config import AppConfig
from app.policy import ZonePolicy
from app.sink import JsonlEventSink
from app.zone_events import watch_zone

def main() -> None:
    config = AppConfig()
    model = Fotonet(config.checkpoint)
    zone = BoxTransform(config.zone_xywh, image_size=config.image_size)
    policy = ZonePolicy()
    sink = JsonlEventSink(config.events)
    for event in watch_zone(model, config.source, zone, policy):
        sink.append(event)
        print(event)

if __name__ == "__main__":
    main()`, "Compose the complete zone-event process",
          "The entry point owns construction only: Fotonet produces tracks, the geometry module detects transitions, and the sink persists them.",
          "This five-file boundary is small enough to copy yet realistic enough to replace individual adapters in a deployed service."),
      ],
    },
  ],

  models: [
    {
      tier: 1,
      title: "List and size canonical graphs",
      summary: "Enumerate public model names and count parameters without training or weights.",
      files: [
        py("list_models.py", `from fotonet import Fotonet

for name in Fotonet.MODELS:
    print(name)`, "Use the registry as the public source",
          "The explicit registry defines published model names. A YAML file existing on disk does not automatically make it public.",
          "Canonical names cover N, S, M, L, and X with optional P2 variants."),
        py("count_parameters.py", `from fotonet import Fotonet

for name in Fotonet.MODELS:
    wrapper = Fotonet(name)
    total = sum(parameter.numel() for parameter in wrapper.model.parameters())
    trainable = sum(parameter.numel() for parameter in wrapper.model.parameters() if parameter.requires_grad)
    print(f"{name:12} total={total:,} trainable={trainable:,}")`, "Count the graph you actually constructed",
          "Parameter counts describe storage and capacity, not accuracy. Full training graphs can also contain branches removed from slim deployment checkpoints.",
          "Use the benchmark methodology for comparable MAC, latency, throughput, and memory numbers."),
      ],
    },
    {
      tier: 2,
      title: "Inspect output contracts",
      summary: "Run synthetic inputs through sibling graphs and compare raw candidate shapes.",
      files: [
        py("output_shapes.py", `import torch
from fotonet import Fotonet

for name in ("fotonetn", "fotonetn-p2"):
    wrapper = Fotonet(name, nc=80)
    wrapper.model.eval()
    with torch.inference_mode():
        output = wrapper.model(torch.zeros(1, 3, 640, 640, device=wrapper.device))
    print(name, tuple(output.shape))`, "Observe the raw graph boundary",
          "The stable exported layout is [B, N, nc + 4]. P2 increases N because it adds a stride-4 feature level.",
          "A zero tensor is suitable for shape inspection only; it says nothing about detection quality."),
        py("module_inventory.py", `from collections import Counter
from fotonet import Fotonet

for name in ("fotonetn", "fotonetn-p2"):
    model = Fotonet(name).model
    counts = Counter(type(module).__name__ for module in model.modules())
    print(name)
    for module_name, count in counts.most_common():
        print(f"  {module_name:24} {count}")`, "Compare topology without reading private state keys",
          "A module inventory gives a stable high-level view and helps explain parameter or operation-count differences.",
          "Do not build application logic around private module names; checkpoint graph identity is the compatibility contract."),
        py("compare_candidates.py", `import torch
from fotonet import Fotonet

def candidates(name: str, imgsz: int = 640) -> int:
    wrapper = Fotonet(name, nc=80)
    wrapper.model.eval()
    with torch.inference_mode():
        output = wrapper.model(torch.zeros(1, 3, imgsz, imgsz, device=wrapper.device))
    return output.shape[1]

base = candidates("fotonetn")
p2 = candidates("fotonetn-p2")
print({"base": base, "p2": p2, "multiplier": p2 / base})`, "Quantify the P2 output expansion",
          "Candidate count exposes the output and postprocessing traffic introduced by the stride-4 level.",
          "This structural comparison still cannot decide whether P2 improves small-object accuracy on a target dataset."),
      ],
    },
    {
      tier: 3,
      title: "Graph profiler harness",
      summary: "Measure warmed CUDA forwards with synchronized timing and an explicit environment record.",
      files: [
        py("bench/profile.py", `import statistics
import time
import torch
from fotonet import Fotonet
from bench.settings import BenchSettings

def profile(settings: BenchSettings) -> dict[str, float | int | str]:
    device = torch.device(settings.device)
    model = Fotonet(settings.model, device=device).model.eval()
    sample = torch.zeros(settings.batch, 3, settings.imgsz, settings.imgsz, device=device)
    with torch.inference_mode():
        for _ in range(settings.warmups):
            model(sample)
        torch.cuda.synchronize()
        samples = []
        for _ in range(settings.repeats):
            start = time.perf_counter()
            model(sample)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    median_ms = statistics.median(samples)
    return {"model": settings.model, "batch": settings.batch,
            "p50_ms": median_ms, "images_per_second": settings.batch * 1000 / median_ms}`, "Synchronize the device around timing",
          "CUDA launches are asynchronous; unsynchronized wall-clock timing mostly measures dispatch.",
          "Record GPU, driver, PyTorch, CUDA, cuDNN, precision, graph state, input shape, warmups, repeats, and summary statistic with every published number."),
        py("bench/environment.py", `import platform
import torch

def environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }`, "Make the benchmark environment reproducible",
          "Latency numbers are backend- and hardware-specific. Environment metadata prevents them from becoming context-free marketing claims.",
          "Measure end-to-end application latency separately from raw model forwards."),
        py("bench/settings.py", `from dataclasses import dataclass

@dataclass(frozen=True)
class BenchSettings:
    model: str = "fotonetn"
    device: str = "cuda:0"
    imgsz: int = 640
    batch: int = 1
    warmups: int = 30
    repeats: int = 100`, "Freeze the benchmark protocol",
          "One settings object makes the timed graph, shape, warmup, and repeat count part of the result identity.",
          "Run changed settings as separate records rather than silently overwriting the protocol."),
        py("bench/report.py", `import json
from pathlib import Path
from typing import Any

def write_report(path: Path, environment: dict[str, Any], results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"environment": environment, "results": results}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")`, "Write one self-describing artifact",
          "Environment and results live together so a latency row cannot be separated from its hardware and software context.",
          "The report is suitable for version control or later comparison; it is not an AP or quality evaluation."),
        py("run_bench.py", `from pathlib import Path
from bench.environment import environment
from bench.profile import profile
from bench.report import write_report
from bench.settings import BenchSettings

def main() -> None:
    settings = [
        BenchSettings(model="fotonetn", batch=1),
        BenchSettings(model="fotonetn", batch=8),
        BenchSettings(model="fotonetn-p2", batch=1),
    ]
    results = [profile(item) for item in settings]
    write_report(Path("outputs/benchmark.json"), environment(), results)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()`, "Run a controlled benchmark matrix",
          "The entry point profiles the same model under explicit batch choices and keeps P2 as a separate graph row.",
          "For publication, keep clocks and background load controlled and report distributions rather than a single best sample."),
      ],
    },
  ],

  checkpoints: [
    {
      tier: 1,
      title: "Safe checkpoint inventory",
      summary: "Inspect tensor-only metadata and calculate an immutable file checksum.",
      files: [
        py("inspect.py", `import torch
from pathlib import Path

path = Path("weights/fotonetn.pt")
payload = torch.load(path, map_location="cpu", weights_only=True)
print("keys", sorted(payload))
for key in ("format_version", "model_name", "nc", "names"):
    if key in payload:
        print(key, payload[key])`, "Use tensor-only loading for inspection",
          "weights_only=True avoids general pickle execution and is the default trust boundary expected by native loading.",
          "Do not set allow_unsafe=True for downloaded or otherwise untrusted legacy files."),
        py("checksum.py", `import hashlib
from pathlib import Path

path = Path("weights/fotonetn.pt")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"{digest}  {path.name}")`, "Identify bytes, not only a filename",
          "Release weights should publish SHA256 checksums. A renamed or replaced file then fails verification.",
          "A checksum proves identity and integrity, not model quality or provenance on its own."),
      ],
    },
    {
      tier: 2,
      title: "Full versus slim audit",
      summary: "Compare checkpoint capabilities before choosing resume or pretrained behavior.",
      files: [
        py("classify.py", `import torch
from pathlib import Path

def classify(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    resumable = all(key in payload for key in (
        "optimizer", "scheduler", "epoch", "rng_state"
    ))
    return "full-resume" if resumable else "slim-inference"

for path in Path("weights").glob("*.pt"):
    print(path.name, classify(path))`, "Inspect capability rather than guessing from names",
          "A last checkpoint is expected to retain training state; an inference-only save intentionally removes it.",
          "Use the package loader for authoritative compatibility validation. This small script is an inventory aid."),
        py("load_modes.py", `from fotonet import Fotonet

# Inference: either full or slim native checkpoint.
inference_model = Fotonet("weights/fotonetn-inference.pt")

# Fresh optimizer state from compatible weights.
fresh_model = Fotonet("fotonetn", nc=80)
fresh_model.train(
    data="data.yaml",
    weights="weights/fotonetn-inference.pt",
    pretrained=True,
    save_dir="runs/new-run",
)

# Exact continuation requires the full last checkpoint.
resume_model = Fotonet("runs/run-a/fotonet_last.pt")
resume_model.train(data="data.yaml", resume=True, save_dir="runs/run-a")`, "Choose one of three distinct operations",
          "Loading for inference, starting fresh from weights, and exact resume have different state requirements.",
          "resume=True and pretrained=True are mutually exclusive because they make contradictory claims about optimizer continuity."),
      ],
    },
    {
      tier: 3,
      title: "Atomic checkpoint handoff",
      summary: "Verify a generated inference checkpoint before making it visible to downstream jobs.",
      files: [
        py("tools/publish_local.py", `from pathlib import Path
from fotonet import Fotonet

def publish_inference(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.pending")
    model = Fotonet(source)
    model.save(temporary, inference_only=True, half=False)
    Fotonet(temporary)  # reconstruct and validate before handoff
    temporary.replace(target)
    return target`, "Validate before the atomic rename",
          "Writing under a pending name prevents consumers from observing a partially written final path.",
          "The final replace is atomic only within one filesystem. Remote object stores need their own staged publication contract."),
        py("tools/verify_schema.py", `from pathlib import Path
from fotonet import Fotonet

def verify_schema(checkpoint: Path, expected: list[str]) -> None:
    model = Fotonet(checkpoint)
    actual = [model.names[index] for index in range(model.nc)]
    if actual != expected:
        raise ValueError(f"class schema mismatch: {actual!r}")`, "Check semantic output identity",
          "Equal class counts do not imply equal meanings. Ordered class names determine how output indices are interpreted.",
          "For an 80-class checkpoint, compare the complete schema rather than this shortened illustrative list."),
        py("tools/inventory.py", `import hashlib
from pathlib import Path
import torch

def inventory(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "format_version": payload.get("format_version"),
        "model_name": payload.get("model_name"),
        "nc": payload.get("nc"),
    }`, "Inventory the exact published bytes",
          "Tensor-only loading captures checkpoint identity fields without enabling arbitrary pickle execution.",
          "The SHA256 belongs in release assets or notes beside the checkpoint, not only in a local log."),
        py("release_policy.py", `from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ReleasePolicy:
    source: Path
    target: Path
    expected_names: list[str]
    manifest: Path = Path("weights/checksums.json")

    def validate(self) -> None:
        if self.source == self.target:
            raise ValueError("source and target must be different")
        if not self.expected_names:
            raise ValueError("ordered class names are required")`, "Define the handoff before touching files",
          "The policy binds source, destination, ordered class schema, and manifest path into one validated release operation.",
          "A remote publisher can reuse the same policy after adding credentials and object-store staging outside this example."),
        py("release_checkpoint.py", `import json
from pathlib import Path
from release_policy import ReleasePolicy
from tools.inventory import inventory
from tools.publish_local import publish_inference
from tools.verify_schema import verify_schema

def release(policy: ReleasePolicy) -> dict[str, object]:
    policy.validate()
    published = publish_inference(policy.source, policy.target)
    verify_schema(published, policy.expected_names)
    record = inventory(published)
    policy.manifest.parent.mkdir(parents=True, exist_ok=True)
    policy.manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record

if __name__ == "__main__":
    release(ReleasePolicy(
        source=Path("runs/run-a/fotonet_last.pt"),
        target=Path("weights/fotonetn-inference.pt"),
        expected_names=["person", "bicycle", "car"],
    ))`, "Orchestrate a verified local release",
          "Publication proceeds from full checkpoint to staged inference save, reconstruction, schema validation, checksum inventory, and manifest.",
          "The three-name schema is illustrative; real releases must provide the complete ordered class list and publish the manifest with the weight."),
      ],
    },
  ],

  api: [
    {
      tier: 1,
      title: "Results object basics",
      summary: "Use the public facade and iterate the two collection levels correctly.",
      files: [
        py("results_basics.py", `from fotonet import Fotonet

model = Fotonet("weights/fotonetn.pt")
results = model.predict(["a.jpg", "b.jpg"], conf=0.25)

print("images", len(results))
for result in results:
    print("detections", len(result))
    for box in result.boxes:
        print(box.cls_id, box.cls, box.conf, box.xyxy)`, "Distinguish image results from detections",
          "The outer list is aligned with inputs. Each Results object then owns a DetectionBoxes collection.",
          "box.xywh is normalized; box.xyxy follows the active transform and is suitable for geometry operations."),
        py("outputs.py", `from pathlib import Path

def save_result(result, stem: str) -> None:
    root = Path("outputs")
    root.mkdir(exist_ok=True)
    result.save(root / f"{stem}.jpg")
    (root / f"{stem}.json").write_text(result.to_json() + "\\n")`, "Keep rendered and structured outputs together",
          "The helper accepts a Results object rather than reaching back into model internals.",
          "Pixels must have been retained for save(); path-based image prediction does this by default."),
      ],
    },
    {
      tier: 2,
      title: "Typed application adapter",
      summary: "Translate DetectionBox objects into a small application-owned record.",
      files: [
        py("records.py", `from dataclasses import dataclass
from fotonet import DetectionBox

@dataclass(frozen=True)
class DetectionRecord:
    class_id: int
    name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None

def adapt(box: DetectionBox) -> DetectionRecord:
    return DetectionRecord(
        class_id=box.cls_id,
        name=box.cls,
        confidence=box.conf,
        xyxy=box.xyxy,
        track_id=box.track_id,
    )`, "Own the boundary your application depends on",
          "The adapter isolates downstream code from fields it does not use and gives static tooling a concrete record type.",
          "Preserve normalized-versus-pixel units in names or types when both exist in the same system."),
        py("filters.py", `from collections.abc import Iterable
from fotonet import DetectionBox

def accepted(
    boxes: Iterable[DetectionBox],
    *,
    classes: set[str],
    minimum_confidence: float,
):
    for box in boxes:
        if box.cls in classes and box.conf >= minimum_confidence:
            yield box`, "Keep business filtering outside inference",
          "The model-level confidence threshold limits returned candidates; this second policy can express class-specific application choices.",
          "Do not use a deployment threshold when producing score-ranked evaluation reports."),
        py("policy.py", `from dataclasses import dataclass
from records import DetectionRecord

@dataclass(frozen=True)
class DetectionPolicy:
    classes: frozenset[str]
    minimum_confidence: float = 0.25

    def accepts(self, record: DetectionRecord) -> bool:
        return record.name in self.classes and record.confidence >= self.minimum_confidence`, "Make application filtering portable",
          "The policy consumes the application-owned record rather than a model-internal tensor or module.",
          "Class-specific thresholds can be added here without changing inference or the Results adapter."),
      ],
    },
    {
      tier: 3,
      title: "Protocol-based pipeline",
      summary: "Define a narrow prediction interface so native and exported runtimes can share application code.",
      files: [
        py("app/contracts.py", `from typing import Any, Protocol
from fotonet import Results

class Detector(Protocol):
    def predict(
        self,
        source: Any,
        imgsz: int | tuple[int, int] | None = None,
        conf: float = 0.25,
        max_det: int = 300,
        **kwargs: Any,
    ) -> list[Results]: ...

def detect_one(detector: Detector, source: Any) -> Results:
    results = detector.predict(source, conf=0.25, max_det=100)
    if len(results) != 1:
        raise ValueError(f"expected one input result, got {len(results)}")
    return results[0]`, "Depend on the behavior you use",
          "A structural Protocol keeps the application testable without subclassing Fotonet or exposing graph internals.",
          "Native checkpoints and supported runtime artifacts use the same facade, so most application code should not branch on backend."),
        py("tests/fake_detector.py", `import numpy as np
import torch
from fotonet import Results

class EmptyDetector:
    def predict(self, source, **kwargs):
        return [Results(
            orig_img=np.zeros((64, 64, 3), dtype=np.uint8),
            boxes=torch.empty((0, 4)),
            scores=torch.empty(0),
            classes=torch.empty(0, dtype=torch.long),
            names={},
        )]
`, "Test empty-result behavior without a checkpoint",
          "An application should handle zero detections as a normal result, not an exception.",
          "This fake targets the public Results constructor. Keep fixtures small and avoid representing smoke tests as model-quality evidence."),
        py("app/domain.py", `from dataclasses import dataclass

@dataclass(frozen=True)
class DetectionSummary:
    source_id: str
    detections: int
    classes: tuple[str, ...]

    @classmethod
    def from_result(cls, source_id: str, result) -> "DetectionSummary":
        names = tuple(sorted({box.cls for box in result.boxes}))
        return cls(source_id=source_id, detections=len(result.boxes), classes=names)`, "Keep the application record independent",
          "The domain type stores only fields this service needs and stays serializable without exposing tensors or graph state.",
          "A larger application can version this record independently from the model checkpoint format."),
        py("app/service.py", `from pathlib import Path
from app.contracts import Detector, detect_one
from app.domain import DetectionSummary

class DetectionService:
    def __init__(self, detector: Detector):
        self.detector = detector

    def inspect(self, source: str | Path) -> DetectionSummary:
        path = Path(source)
        result = detect_one(self.detector, path)
        return DetectionSummary.from_result(path.name, result)`, "Put the public API behind one service boundary",
          "The service accepts any object satisfying Detector, so native checkpoints, exported runtimes, and test doubles use the same orchestration.",
          "Source identity and domain adaptation happen once rather than being repeated across HTTP, CLI, or batch entry points."),
        py("tests/test_service.py", `from app.service import DetectionService
from tests.fake_detector import EmptyDetector

def test_empty_result_is_a_valid_summary() -> None:
    summary = DetectionService(EmptyDetector()).inspect("empty.jpg")
    assert summary.source_id == "empty.jpg"
    assert summary.detections == 0
    assert summary.classes == ()`, "Test application behavior without model weights",
          "The fake crosses the same Detector protocol boundary as Fotonet while making the zero-detection case deterministic.",
          "Add a separate integration test with a trusted tiny fixture checkpoint; keep AP evaluation in the canonical validation workflow."),
      ],
    },
  ],

  transforms: [
    {
      tier: 1,
      title: "Anchor-preserving crop",
      summary: "Create a portrait crop while keeping the detection's bottom point stable.",
      files: [
        py("crop_person.py", `from fotonet import AnchorPoint, Fotonet

model = Fotonet("weights/fotonetn.pt")
result = model.predict("street.jpg", retain_images=True)[0]

for index, box in enumerate(result.boxes):
    if box.cls != "person":
        continue
    region = box.transform
    region.set_anchor(AnchorPoint.BOTTOM)
    fixed = region.position
    region.pixel_expand(24)
    region.set_aspect_ratio((4, 5), mode=1)
    moved = region.position
    region.move((fixed.x - moved.x, fixed.y - moved.y))
    region.clamp()
    crop = region.crop(result.orig_img)
    crop.save(f"outputs/person-{index:03}.jpg")`, "Apply geometry before extracting pixels",
          "The bottom anchor represents a floor-contact intent. Padding and aspect fitting expand around the chosen region, then clamp aligns it with image bounds.",
          "Use a fresh transform for independent variants because transform operations are intentionally mutable."),
        py("manual_region.py", `from PIL import Image
from fotonet import AnchorPoint, BoxTransform

image = Image.open("image.jpg").convert("RGB")
region = BoxTransform((0.52, 0.48, 0.30, 0.62), image.size)
region.set_anchor(AnchorPoint.BOTTOM)
fixed = region.position
region.pixel_expand(28).set_aspect_ratio((4, 5), mode=1)
moved = region.position
region.move((fixed.x - moved.x, fixed.y - moved.y)).clamp()
region.crop(image).save("outputs/manual-card.jpg")`, "Use the transform layer without a model",
          "BoxTransform accepts normalized center-format xywh plus image_size in width-height order.",
          "This makes geometry policies testable with known boxes before detection uncertainty is introduced."),
      ],
    },
    {
      tier: 2,
      title: "Containment and overlap policy",
      summary: "Compare contact-point, area-overlap, and pairwise-IoU decisions explicitly.",
      files: [
        py("zone_policy.py", `from dataclasses import dataclass
from fotonet import AnchorPoint, BoxTransform, DetectionBox

@dataclass(frozen=True)
class ZoneDecision:
    anchor_inside: bool
    area_inside_75: bool

def decide(box: DetectionBox, zone: BoxTransform) -> ZoneDecision:
    transform = box.transform.set_anchor(AnchorPoint.BOTTOM)
    anchor_inside = zone.contains(point=transform.position)
    x1, y1, x2, y2 = transform.xyxy
    area_inside_75 = transform.contains(
        x=(zone.xyxy[0], zone.xyxy[2]),
        y=(zone.xyxy[1], zone.xyxy[3]),
        threshold=0.75,
    )
    return ZoneDecision(anchor_inside, area_inside_75)
`, "Return both semantics instead of hiding the choice",
          "A contact point inside a zone and most of a rectangle inside a zone are different claims.",
          "The caller can select point containment or a 75% area threshold appropriate to entrances, shelves, safety regions, or occupancy reporting."),
        py("relationships.py", `def describe(first, second) -> dict[str, float | bool]:
    a = first.transform
    b = second.transform
    return {
        "overlaps": a.overlaps(b),
        "iou": a.iou(b),
        "anchor_distance": a.distance(b.position),
        "pixel_anchor_distance": a.pixel_distance(b.pixel_position),
    }`, "Name coordinate units at the boundary",
          "Normalized distance is resolution-independent; pixel distance is appropriate when a physical image threshold is intended.",
          "IoU measures shared area relative to union and should not be confused with the directional containment fraction."),
      ],
    },
    {
      tier: 3,
      title: "Stateful zone-event engine",
      summary: "An 11-file transform-first application with geometry, state, persistence, packaging, and tests.",
      downloadHref: "/examples/projects/transform-zone-system.zip",
      files: transformSystemFiles.length ? transformSystemFiles : [
        py("app/event_engine.py", `from dataclasses import dataclass
from fotonet import AnchorPoint, BoxTransform

@dataclass(frozen=True)
class Transition:
    frame: int
    track_id: int
    state: str

class ZoneEngine:
    def __init__(self, zone: BoxTransform):
        self.zone = zone
        self.states: dict[int, bool] = {}

    def update(self, frame: int, box) -> Transition | None:
        if box.track_id is None:
            return None
        point = box.transform.set_anchor(AnchorPoint.BOTTOM).position
        inside = self.zone.contains(point=point)
        previous = self.states.get(box.track_id)
        self.states[box.track_id] = inside
        if previous is None or previous == inside:
            return None
        return Transition(frame, box.track_id, "enter" if inside else "exit")`, "Emit only edge transitions",
          "The engine stores the previous decision per visible track and produces a record only when the boolean state changes.",
          "A production system should define first-observation behavior, disappearance timeout, camera reset, and track-ID reuse explicitly."),
        py("app/event_loop.py", `from collections.abc import Iterator
from fotonet import Fotonet
from app.event_engine import Transition, ZoneEngine

def transitions(
    model: Fotonet,
    source: str,
    engine: ZoneEngine,
    class_name: str = "person",
) -> Iterator[Transition]:
    for frame, result in enumerate(model.track(source, stream=True, persist=True)):
        for box in result.boxes:
            if box.cls != class_name:
                continue
            transition = engine.update(frame, box)
            if transition is not None:
                yield transition`, "Keep inference, geometry, and state composable",
          "The event loop does orchestration only. Zone semantics live in the engine and detection production remains in Fotonet.",
          "This separation supports unit tests with manually constructed boxes and integration tests with short recorded clips."),
        py("app/config.py", `from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ZoneConfig:
    checkpoint: Path = Path("weights/fotonetn.pt")
    source: str = "entrance.mp4"
    output: Path = Path("outputs/transitions.jsonl")
    image_size: tuple[int, int] = (1920, 1080)
    zone_xywh: tuple[float, float, float, float] = (0.50, 0.65, 0.55, 0.42)
    class_name: str = "person"`, "Keep geometry and model choices explicit",
          "Normalized zone coordinates are bound to the declared image size, checkpoint, source, and target class.",
          "Validate externally loaded configuration before constructing BoxTransform in a long-running service."),
        py("app/sink.py", `import json
from dataclasses import asdict
from pathlib import Path
from app.event_engine import Transition

class TransitionSink:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, transition: Transition) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            print(json.dumps(asdict(transition), sort_keys=True), file=stream)`, "Persist typed transition records",
          "The sink owns serialization and append behavior, leaving geometry code unaware of files or transport.",
          "Replace this adapter with a queue producer when delivery guarantees matter, while preserving the Transition schema."),
        py("main.py", `from fotonet import BoxTransform, Fotonet
from app.config import ZoneConfig
from app.event_engine import ZoneEngine
from app.event_loop import transitions
from app.sink import TransitionSink

def main() -> None:
    config = ZoneConfig()
    model = Fotonet(config.checkpoint)
    zone = BoxTransform(config.zone_xywh, image_size=config.image_size)
    engine = ZoneEngine(zone)
    sink = TransitionSink(config.output)
    for transition in transitions(model, config.source, engine, config.class_name):
        sink.write(transition)
        print(transition)

if __name__ == "__main__":
    main()`, "Run transforms as the system's decision boundary",
          "The complete application treats AnchorPoint and BoxTransform decisions as the central logic, with detection upstream and persistence downstream.",
          "Engine, loop, configuration, sink, and entry point are independently replaceable and connected through small typed values."),
        py("tests/test_event_engine.py", `from fotonet import BoxTransform
from app.event_engine import ZoneEngine

class StubTransform:
    def __init__(self, point):
        self.position = point

    def set_anchor(self, anchor):
        return self

class StubBox:
    track_id = 7
    transform = StubTransform((0.5, 0.5))

def test_first_observation_does_not_emit_transition() -> None:
    zone = BoxTransform((0.5, 0.5, 0.5, 0.5), image_size=(100, 100))
    assert ZoneEngine(zone).update(0, StubBox()) is None`, "Lock the first-observation rule",
          "A newly visible track establishes state but does not pretend an entrance happened inside the camera view.",
          "Add inside-to-outside and outside-to-inside cases with real Vector2 fixtures when integrating this example."),
        py("tests/test_config.py", `from app.config import ZoneConfig

def test_default_zone_is_normalized() -> None:
    config = ZoneConfig()
    cx, cy, width, height = config.zone_xywh
    assert 0 <= cx <= 1 and 0 <= cy <= 1
    assert 0 < width <= 1 and 0 < height <= 1
    assert config.class_name`, "Fail fast on invalid geometry defaults",
          "This small contract test protects normalized coordinate assumptions before video or model resources are opened.",
          "External configuration loaders should apply the same bounds and also verify file paths and supported image sizes."),
      ],
    },
  ],
};
