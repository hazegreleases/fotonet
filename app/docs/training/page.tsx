import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { ExampleRepositories } from "../../ui/ExampleRepositories";
import { launcherCode, resumeCode, trainCode } from "../../data";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "Train and Resume a PyTorch Object Detector",
  description: "Train fotonet on a custom object detection dataset, configure optimization and validation, save complete checkpoints, and resume interrupted PyTorch runs.",
  path: "/docs/training",
  keywords: ["train object detector", "resume PyTorch training", "custom object detection", "detection checkpoint"],
});

export default function TrainingPage() {
  return (
    <>
      <DocHeader eyebrow="How-to / training" title="Train the production graph and resume the same run." lead="The dedicated launcher is locked to the production V1 detector, uniform sampling, and the exact run’s full last checkpoint." />
      <section className="doc-section prose">
        <h2 id="dataset">Dataset contract</h2>
        <p>YOLO label rows contain normalized values:</p>
        <CodeBlock code={`class_id x_center y_center width height`} language="text" label="labels/image-name.txt" />
        <CodeBlock code={`path: /path/to/dataset\ntrain: images/train\nval: images/val\nnc: 2\nnames:\n  0: cat\n  1: dog`} language="yaml" label="data.yaml" />
        <p>The validation source must be explicit and independent. The default annotation policy fixes valid edge crossings, removes malformed or duplicate rows, and records missing labels as background images. Use <code>annotation_policy=&quot;error&quot;</code> for a strict audit.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="launcher">Architecture-specific launcher</h2>
        <CodeBlock code={launcherCode} language="bash" label="Fresh run" />
        <p>The launcher accepts only the ten canonical N/S/M/L/X and P2 names and refuses removed training systems. It binds the run to the graph, ordered class schema, dataset hash, and uniform policy before training begins.</p>
        <CodeBlock code={`python train.py --model fotonetn --data path/to/data.yaml --run-dir runs/fotonetn --dry-run`} language="bash" label="Validate the plan without constructing a dataset or optimizer" />
      </section>
      <section className="doc-section prose">
        <h2 id="resume">Resume after interruption</h2>
        <CodeBlock code={resumeCode} language="bash" label="Resume the same run" />
        <p>Repeat the same model, data, epoch total, batch settings, and run directory. Bare <code>--resume</code> resolves only to that directory’s <code>fotonet_last.pt</code>.</p>
        <Note title="Resume is continuation, not initialization" tone="warning">
          <p>The launcher requires matching run identity, graph fingerprint, class order, uniform sampling policy, model and EMA O2M state, optimizer, scheduler, scaler, RNG, and step accounting. It will not search for a “latest” run or fall back to a best checkpoint.</p>
        </Note>
      </section>
      <section className="doc-section prose">
        <h2 id="python">General Python API</h2>
        <CodeBlock code={trainCode} label="Python" />
        <p>The general API remains available for controlled experiments. For reproducible production-graph runs and strict interruption recovery, prefer the dedicated launcher above.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="checkpoint-modes">Checkpoint modes</h2>
        <dl className="definition-list">
          <div><dt><code>resume=True</code></dt><dd>Continue one full checkpoint with optimizer, scheduler, scaler, EMA, RNG, uniform-sampling identity, and epoch state.</dd></div>
          <div><dt><code>pretrained=True</code></dt><dd>Load compatible weights and begin fresh optimizer/scheduler state.</dd></div>
          <div><dt><code>fotonet_last.pt</code></dt><dd>Resumable full training checkpoint.</dd></div>
          <div><dt><code>fotonet_best.pt</code></dt><dd>May be a stripped inference checkpoint and is not a resume source.</dd></div>
        </dl>
      </section>
      <section className="doc-section prose">
        <h2 id="metrics">Validation protocol</h2>
        <p><code>val_conf=0.0</code> retains finite detections so COCO’s score-ranked <code>maxDets</code> determines AP. Deployment-point precision and recall use separate <code>operating_conf</code> and <code>operating_iou</code> settings.</p>
        <Note title="No AP by smoke test"><p>Only a released checkpoint, declared split, exact protocol, and reproducible command support a public accuracy claim.</p></Note>
      </section>
      <ExampleRepositories topic="training" />
      <NextLinks items={[
        { href: "/docs/checkpoints", label: "Checkpoint modes", detail: "Full resume state, slim inference files, and identity" },
        { href: "/docs/validation", label: "Validate the run", detail: "COCO ranking, operating points, and release evidence" },
      ]} />
    </>
  );
}
