import type { Metadata } from "next";
import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";

export const metadata: Metadata = { title: "Command line" };

export default function CliPage() {
  return (
    <>
      <DocHeader eyebrow="Reference / CLI" title="Five tasks, explicit models, key=value options." lead="The command line mirrors the Python workflows for training, prediction, tracking, validation, and export without alias spellings or implicit model selection." />
      <section className="doc-section prose">
        <h2 id="syntax">Syntax</h2>
        <CodeBlock code={`fotonet <task> model=<model-or-checkpoint> key=value ...`} language="bash" label="General form" />
        <p>Every task requires <code>model=</code>. Training may use a canonical model ID; prediction, tracking, validation, and export normally use an explicit checkpoint or exported artifact.</p>
        <Note title="Argument spelling"><p>Use <code>key=value</code>. Hyphenated keys normalize to underscores, booleans accept <code>true</code>/<code>false</code>, and image size accepts an integer or pair such as <code>imgsz=(640,960)</code>.</p></Note>
      </section>
      <section className="doc-section prose">
        <h2 id="tasks">Task examples</h2>
        <dl className="api-list">
          <div><dt><code>predict</code></dt><dd><CodeBlock code={`fotonet predict model=weights/fotonetn.pt source=image.jpg conf=0.25 save=true`} language="bash" label="Predict" /></dd></div>
          <div><dt><code>track</code></dt><dd><CodeBlock code={`fotonet track model=weights/fotonetn.pt source=video.mp4 stream=true persist=true`} language="bash" label="Track" /></dd></div>
          <div><dt><code>val</code></dt><dd><CodeBlock code={`fotonet val model=weights/fotonetn.pt data=data.yaml imgsz=640 batch=8`} language="bash" label="Validate" /></dd></div>
          <div><dt><code>export</code></dt><dd><CodeBlock code={`fotonet export model=weights/fotonetn.pt format=onnx path=exports/fotonet.onnx`} language="bash" label="Export" /></dd></div>
          <div><dt><code>train</code></dt><dd><CodeBlock code={`fotonet train model=fotonetn data=data.yaml epochs=100 batch=16 imgsz=640`} language="bash" label="General training API" /></dd></div>
        </dl>
      </section>
      <section className="doc-section prose">
        <h2 id="launcher">Strict training launcher</h2>
        <p>Download the public <code>train.py</code> launcher for friendlier options, a printed resolved plan, dry-run validation, bare or explicit resume, pretrained initialization, and validated <code>--set key=value</code> forwarding.</p>
        <p><a className="download-button" href="/examples/train.py" download>Download public train.py <span aria-hidden="true">↓</span></a></p>
        <CodeBlock code={`python train.py --model fotonetn --data data.yaml --epochs 300 --batch 16 --run-dir runs/fotonetn --resume`} language="bash" label="Resume" />
      </section>
      <NextLinks items={[
        { href: "/docs/api", label: "Python API", detail: "The same workflows with typed result objects" },
        { href: "/docs/examples", label: "Runnable scripts", detail: "Download complete programs rather than command fragments" },
      ]} />
    </>
  );
}
