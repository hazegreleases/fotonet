import type { Metadata } from "next";
import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";

export const metadata: Metadata = { title: "Validation" };

export default function ValidationPage() {
  return (
    <>
      <DocHeader eyebrow="How-to / evaluation" title="Separate model ranking from a deployment threshold." lead="Validation keeps score-ranked COCO evaluation, an operating-point report, annotation policy, and release evidence distinct." />
      <section className="doc-section prose">
        <h2 id="python">Validate from Python</h2>
        <CodeBlock code={`from fotonet import Fotonet

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
print(metrics["mAP50_95"])`} label="Python" />
        <p><code>conf=0.0</code> retains finite candidates for score ranking; COCO <code>maxDets</code> then limits detections. <code>operating_conf</code> and <code>operating_iou</code> report a separate practical threshold.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="cli">CLI validation</h2>
        <CodeBlock code={`fotonet val model=weights/fotonetn.pt data=data.yaml imgsz=640 batch=8 conf=0.0 max_det=100`} language="bash" label="Terminal" />
        <p>The validation dataset must be explicit and compatible with the checkpoint’s ordered class schema.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="release">Canonical COCO release validation</h2>
        <p>The release process first performs a non-evaluating identity and dataset preflight, then runs the official COCO val2017 protocol at 640×640 with 100 maximum detections, all 5,000 validation images, and the canonical 80-category mapping.</p>
        <Note title="What supports a public AP claim" tone="warning"><p>A claim must identify the released checkpoint and checksum, dataset split, image size, max-detections policy, evaluator backend, and reproducible command. Smoke tests and architecture inspection do not qualify.</p></Note>
      </section>
      <section className="doc-section prose">
        <h2 id="interpret">Read the output carefully</h2>
        <dl className="definition-list">
          <div><dt><code>mAP50_95</code></dt><dd>Mean AP across IoU thresholds 0.50 through 0.95.</dd></div>
          <div><dt><code>mAP50</code></dt><dd>Average precision at IoU 0.50.</dd></div>
          <div><dt>Operating precision/recall</dt><dd>Thresholded deployment-point values, not substitutes for ranked AP.</dd></div>
          <div><dt>Sanitized annotations</dt><dd>Metrics must be labeled noncanonical when the annotation policy changes the evaluated data.</dd></div>
        </dl>
      </section>
      <NextLinks items={[
        { href: "/docs/data", label: "Dataset contract", detail: "Class order, validation sources, labels, and annotation policy" },
        { href: "/benchmarks", label: "Deployment measurements", detail: "Runtime cost without an accuracy implication" },
      ]} />
    </>
  );
}
