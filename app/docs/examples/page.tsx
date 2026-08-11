import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { ExampleRepositories } from "../../ui/ExampleRepositories";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "Runnable Python Object Detection Examples",
  description: "Download complete fotonet scripts for inference, training, validation, ONNX export, AnchorPoint transforms, detection crops, focused subregions, and zone gating.",
  path: "/docs/examples",
  keywords: ["object detection Python example", "train.py", "ONNX export example", "computer vision scripts"],
});

const imageScript = `from argparse import ArgumentParser
from pathlib import Path

from fotonet import Fotonet

parser = ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("image", type=Path)
parser.add_argument("--output", type=Path, default=Path("outputs/prediction.jpg"))
args = parser.parse_args()

model = Fotonet(args.checkpoint)
result = model.predict(args.image, conf=0.25, imgsz=640)[0]
result.save(args.output)
print(result.to_json())`;

const folderScript = `model = Fotonet(args.checkpoint)
results = model.predict(args.images, batch=args.batch, conf=0.25, imgsz=640)

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8") as stream:
    for result in results:
        stream.write(result.to_json() + "\\n")`;

const exportScript = `model = Fotonet(args.checkpoint)
exported = model.export(
    format="onnx",
    path=args.output,
    imgsz=args.imgsz,
    dynamic=args.dynamic,
    verify=True,
)
print(exported["artifact"])
print(exported["metadata"])`;

const validationScript = `model = Fotonet(args.checkpoint)
metrics = model.val(
    data=args.data,
    imgsz=args.imgsz,
    batch=args.batch,
    conf=0.0,
    max_det=100,
)
for name, value in metrics.items():
    if not name.endswith("per_class"):
        print(f"{name}: {value}")`;

const transformRegionCore = `image = Image.open(args.image).convert("RGB")
region = BoxTransform(tuple(args.xywh), image_size=image.size)

region.set_anchor(args.anchor).pixel_expand(args.padding)
if args.aspect:
    fit_aspect_at_anchor(region, args.aspect, mode=1)
region.pixel_move(args.move).clamp()
region.crop(image).save(args.output)`;

const detectionCropCore = `result = model.predict(args.image, conf=args.conf, retain_images=True)[0]

for box in result.boxes:
    if args.class_name and box.cls not in args.class_name:
        continue
    region = box.transform
    if args.focus:
        region.focus(*args.focus)
    region.set_anchor(args.anchor).pixel_expand(args.padding)
    if args.aspect:
        fit_aspect_at_anchor(region, args.aspect)
    region.clamp()
    crop = region.crop(result.orig_img)`;

const zoneFilterCore = `zone = BoxTransform(zone_xywh, image_size=result.image_size).clamp()

for box in result.boxes:
    contact = box.transform.set_anchor(args.anchor).pixel_position
    inside = zone.pixel_contains(point=contact)
    print(box.cls, box.conf, inside, tuple(contact))`;

const trackEventsCore = `previous: dict[int, bool] = {}

for frame_index, result in enumerate(frames):
    zone = zone_transform(zone_pixels, result.image_size)
    for box in result.boxes:
        point = box.transform.set_anchor(AnchorPoint.BOTTOM).pixel_position
        inside = bool(zone.pixel_contains(point=point))
        was_inside = previous.get(box.track_id, False)
        previous[box.track_id] = inside
        if inside != was_inside:
            write_event("enter" if inside else "exit", frame_index, box, point)`;

export default function ExamplesPage() {
  return (
    <>
      <DocHeader eyebrow="Examples / complete files" title="Scripts you can save, understand, and run as written." lead="These are complete command-line programs with validation, output handling, and explicit data contracts—not isolated pseudocode. Use them directly or as application starting points." />
      <section className="example-prerequisites" aria-label="Example prerequisites">
        <div><span>01</span><strong>Install</strong><p><code>python -m pip install fotonet</code></p></div>
        <div><span>02</span><strong>Provide</strong><p>A compatible <code>.pt</code> checkpoint for model workflows. The manual transform example needs only an image.</p></div>
        <div><span>03</span><strong>Run</strong><p>Python 3.10+; add export dependencies for ONNX.</p></div>
      </section>
      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Example 01 / image</p><h2 id="predict-image">Predict one image</h2></div>
          <a className="download-button" href="/examples/predict_image.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>Writes an annotated image and prints the detections as JSON. The output parent directory is created by <code>Results.save()</code>.</p>
        <CodeBlock code={imageScript} label="predict_image.py" />
        <CodeBlock code={`python predict_image.py weights/fotonet.pt images/example.jpg --output outputs/example.jpg`} language="bash" label="Run it" />
      </section>
      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Example 02 / folder</p><h2 id="predict-folder">Batch a folder into JSON Lines</h2></div>
          <a className="download-button" href="/examples/predict_folder.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>Produces one JSON array per source image in input order. This makes the image-to-result mapping explicit while keeping the output streamable.</p>
        <CodeBlock code={folderScript} label="predict_folder.py / core" />
        <CodeBlock code={`python predict_folder.py weights/fotonet.pt images/ --batch 8 --output outputs/detections.jsonl`} language="bash" label="Run it" />
      </section>
      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Example 03 / export</p><h2 id="export-onnx">Export and verify ONNX</h2></div>
          <a className="download-button" href="/examples/export_onnx.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>Creates the ONNX graph and its metadata sidecar. Pass <code>--dynamic</code> only when the consumer needs variable spatial dimensions.</p>
        <CodeBlock code={exportScript} label="export_onnx.py / core" />
        <CodeBlock code={`python export_onnx.py weights/fotonet.pt --output outputs/fotonet.onnx`} language="bash" label="Run it" />
        <Note title="Install the export extra"><p>Use <code>python -m pip install &quot;fotonet[export]&quot;</code> before the ONNX example. TensorRT and CoreML still require their platform-specific toolchains.</p></Note>
      </section>
      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Example 04 / validation</p><h2 id="validate-checkpoint">Validate a checkpoint</h2></div>
          <a className="download-button" href="/examples/validate_checkpoint.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>Runs the public validation protocol against the explicit <code>val</code> source in a dataset YAML and prints aggregate metrics.</p>
        <CodeBlock code={validationScript} label="validate_checkpoint.py / core" />
        <CodeBlock code={`python validate_checkpoint.py weights/fotonet.pt data.yaml --batch 8 --imgsz 640`} language="bash" label="Run it" />
      </section>
      <section className="doc-section prose">
        <div className="example-heading">
          <div><p className="eyebrow">Example 05 / training</p><h2 id="train-current">Train or resume with the public launcher</h2></div>
          <a className="download-button" href="/examples/train.py" download>Download train.py <span aria-hidden="true">↓</span></a>
        </div>
        <p>The public launcher is verbose and approachable: it exposes common optimization, caching, validation, checkpoint, and device controls, plus repeatable <code>--set key=value</code> forwarding for any validated <code>Fotonet.train()</code> kwarg.</p>
        <CodeBlock code={`python train.py --data path/to/data.yaml --model fotonetn --epochs 100 --batch 16 --run-dir runs/fotonetn`} language="bash" label="Fresh run" />
        <CodeBlock code={`python train.py --data path/to/data.yaml --model fotonetn --epochs 100 --batch 16 --run-dir runs/fotonetn --resume`} language="bash" label="Resume the same run" />
        <CodeBlock code={`python train.py --data data.yaml --model fotonetn --optimizer adamw --lr0 0.001 --no-cache-to-ram --set imgsz_schedule='[[0.5,512],[1.0,640]]' -vv`} language="bash" label="Advanced example" />
        <Note title="Preview before training"><p>Add <code>--dry-run</code> to load the model or checkpoint and print every resolved option without constructing a dataset or optimizer.</p></Note>
      </section>
      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Example 06 / transform geometry</p><h2 id="transform-region">Crop a manual region without running a model</h2></div>
          <a className="download-button" href="/examples/transform_region.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>Useful for learning the transform contract, reproducing a crop from stored coordinates, or applying model-independent regions. Input <code>xywh</code> is normalized; padding and movement are source-image pixels. The script prints both original and final geometry as JSON.</p>
        <CodeBlock code={transformRegionCore} label="transform_region.py / core" />
        <CodeBlock code={`python transform_region.py image.jpg --xywh .52 .48 .30 .62 --aspect 4:5 --anchor bottom --padding 28 --output outputs/card.jpg`} language="bash" label="Run it" />
      </section>
      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Example 07 / crop pipeline</p><h2 id="extract-crops">Build a labeled crop directory and manifest</h2></div>
          <a className="download-button" href="/examples/extract_detection_crops.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>Creates confidence-ordered crops with safe filenames and a machine-readable manifest. Repeat <code>--class-name</code> to keep several exact labels. Use <code>--focus 0 0 1 .55</code> to turn each person box into an upper-body region before shaping and padding it.</p>
        <CodeBlock code={detectionCropCore} label="extract_detection_crops.py / core" />
        <CodeBlock code={`python extract_detection_crops.py weights/fotonet.pt street.jpg --class-name person --aspect 4:5 --anchor bottom --padding 24 --output-dir outputs/people`} language="bash" label="Full-object crops" />
        <CodeBlock code={`python extract_detection_crops.py weights/fotonet.pt street.jpg --class-name person --focus 0 0 1 .55 --aspect 1:1 --padding 12 --output-dir outputs/upper-body`} language="bash" label="Focused subregions" />
      </section>
      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Example 08 / application zone</p><h2 id="anchor-zone">Test a floor-contact anchor against a zone</h2></div>
          <a className="download-button" href="/examples/anchor_zone_filter.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>Uses the bottom midpoint instead of the box center to approximate where a person or vehicle touches the ground. It writes an annotated image plus JSON containing every tested anchor and decision. Select another anchor when your scene semantics demand it.</p>
        <CodeBlock code={zoneFilterCore} label="anchor_zone_filter.py / policy" />
        <CodeBlock code={`python anchor_zone_filter.py weights/fotonet.pt entrance.jpg --zone 180 240 940 700 --anchor bottom --class-name person --output outputs/entrance-zone.jpg`} language="bash" label="Run it" />
        <Note title="Anchor tests are not segmentation"><p>A detection anchor is one representative point derived from a rectangle. Use area containment when most of the box must lie inside a region, and use segmentation when exact object boundaries matter.</p></Note>
      </section>
      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Example 09 / event log</p><h2 id="track-zone-events">Log tracked zone transitions from video</h2></div>
          <a className="download-button" href="/examples/track_zone_events.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>Processes an ordered video stream, applies the same bottom-anchor zone policy to each tracked detection, and writes an event only when a visible track changes state. JSON Lines is flushed after each transition, so downstream tools can tail the file while processing continues.</p>
        <CodeBlock code={trackEventsCore} label="track_zone_events.py / transition policy" />
        <CodeBlock code={`python track_zone_events.py weights/fotonet.pt entrance.mp4 --zone 180 240 940 700 --anchor bottom --class-name person --output outputs/zone-events.jsonl`} language="bash" label="Run it" />
        <Note title="Know the tracker boundary"><p>The built-in tracker is greedy, same-class IoU association. Events describe visible rectangle tracks; they do not claim motion prediction, re-identification after long occlusion, or exact segmentation boundaries.</p></Note>
      </section>
      <ExampleRepositories topic="examples" />
      <NextLinks items={[
        { href: "/docs/transforms", label: "Understand spatial transforms", detail: "Anchors, focus, containment, ordering, and relationships" },
        { href: "/docs/validation", label: "Validate a checkpoint", detail: "Ordinary validation and canonical release evidence" },
      ]} />
    </>
  );
}
