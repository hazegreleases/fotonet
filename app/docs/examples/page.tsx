import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "Runnable Python Object Detection Examples",
  description: "Download complete fotonet scripts for image and folder inference, custom object detector training and resume, checkpoint validation, and ONNX export.",
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

export default function ExamplesPage() {
  return (
    <>
      <DocHeader eyebrow="Examples / complete files" title="Scripts you can save and run as written." lead="Each example is a complete command-line program, not pseudocode. Supply a trusted checkpoint and the indicated local input; the scripts create their output directories themselves." />
      <section className="example-prerequisites" aria-label="Example prerequisites">
        <div><span>01</span><strong>Install</strong><p><code>python -m pip install fotonet</code></p></div>
        <div><span>02</span><strong>Provide</strong><p>A compatible <code>.pt</code> checkpoint and local input files.</p></div>
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
      <NextLinks items={[
        { href: "/docs/api", label: "Read the Python API", detail: "Public methods, result classes, and return types" },
        { href: "/docs/validation", label: "Validate a checkpoint", detail: "Ordinary validation and canonical release evidence" },
      ]} />
    </>
  );
}
