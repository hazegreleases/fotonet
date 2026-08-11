import Link from "next/link";
import { pageMetadata } from "../seo";
import { DocHeader, Note } from "../ui/Docs";

export const metadata = pageMetadata({
  title: "fotonet Object Detection Documentation",
  description: "Task-oriented documentation for installing, training, evaluating, exporting, and integrating the fotonet NMS-free PyTorch object detector.",
  path: "/docs",
  keywords: ["object detection documentation", "PyTorch detection library", "computer vision API"],
});

export default function DocsIndex() {
  return (
    <>
      <DocHeader eyebrow="Manual / map" title="Documentation that starts from the job." lead="Choose a path by what you are trying to accomplish. The manual separates first-run guidance, working procedures, factual reference, and architectural explanation." />
      <Note title="Weight release status" tone="warning">
        <p>The first official Nano weight is training now. Until it is released with a SHA256 checksum and canonical validation report, model names construct untrained graphs and inference requires an explicit trusted checkpoint.</p>
      </Note>
      <section className="doc-section">
        <h2 id="learn">Learn the basic workflow</h2>
        <div className="doc-map-list">
          <Link href="/docs/install"><b>01</b><span><strong>Install and first inference</strong><small>Set up Python, construct a model, and load a real checkpoint.</small></span></Link>
          <Link href="/docs/inference"><b>02</b><span><strong>Understand Results</strong><small>Predict images, folders, tensors, BGR frames, and video streams.</small></span></Link>
          <Link href="/docs/cli"><b>03</b><span><strong>Use the command line</strong><small>Run predict, track, train, validation, and export tasks with explicit model inputs.</small></span></Link>
        </div>
      </section>
      <section className="doc-section">
        <h2 id="work">Complete a task</h2>
        <div className="doc-map-list">
          <Link href="/docs/data"><b>04</b><span><strong>Prepare a dataset</strong><small>Define paths, ordered class names, YOLO labels, and annotation policy.</small></span></Link>
          <Link href="/docs/training"><b>05</b><span><strong>Train or resume</strong><small>Use the strict launcher and preserve complete interruption-safe state.</small></span></Link>
          <Link href="/docs/validation"><b>06</b><span><strong>Validate honestly</strong><small>Separate deployment thresholds from ranked COCO evaluation and release claims.</small></span></Link>
          <Link href="/docs/export"><b>07</b><span><strong>Export a deployment graph</strong><small>Choose ONNX, TorchScript, TensorRT, or CoreML and retain metadata.</small></span></Link>
        </div>
      </section>
      <section className="doc-section">
        <h2 id="reference">Look up a contract</h2>
        <div className="doc-map-list">
          <Link href="/docs/models"><b>08</b><span><strong>Models and configuration</strong><small>Scale profiles, P2 variants, fingerprints, and future capacity bands.</small></span></Link>
          <Link href="/docs/checkpoints"><b>09</b><span><strong>Checkpoint contracts</strong><small>Full versus slim files, resume versus pretrained, schemas, and safe loading.</small></span></Link>
          <Link href="/docs/api"><b>10</b><span><strong>Python API reference</strong><small>Public classes, methods, parameters, and return behavior.</small></span></Link>
          <Link href="/docs/transforms"><b>11</b><span><strong>Results and transforms</strong><small>Boxes, coordinates, anchors, crops, movement, containment, and IoU.</small></span></Link>
          <Link href="/docs/examples"><b>12</b><span><strong>Runnable example scripts</strong><small>Complete downloadable files with commands, inputs, and outputs.</small></span></Link>
          <Link href="/benchmarks"><b>13</b><span><strong>Measured performance</strong><small>Parameters, MACs, raw latency, throughput, memory, and exact conditions.</small></span></Link>
        </div>
      </section>
      <section className="doc-section prose">
        <h2 id="invariants">Contracts worth remembering</h2>
        <dl className="definition-list">
          <div><dt>Image prediction</dt><dd>Returns <code>list[Results]</code>, one entry per input.</dd></div>
          <div><dt>Streaming video</dt><dd>Returns an iterator only when <code>stream=True</code>.</dd></div>
          <div><dt>Resume</dt><dd>Requires a full last checkpoint with optimizer, scheduler, scaler, RNG, EMA, graph, class, and run identity.</dd></div>
          <div><dt>Export output</dt><dd><code>[B, N, nc + 4]</code>: class logits followed by normalized <code>xywh</code>.</dd></div>
          <div><dt>Validation</dt><dd>Requires an explicit independent validation source; it never silently uses training data.</dd></div>
        </dl>
      </section>
    </>
  );
}
