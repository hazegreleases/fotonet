import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { ExampleRepositories } from "../../ui/ExampleRepositories";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "Python Object Detection API Reference",
  description: "Reference the fotonet Python API for detector construction, prediction, tracking, training, validation, export, Results objects, and detection transforms.",
  path: "/docs/api",
  keywords: ["object detection API", "Python API reference", "PyTorch detector API", "computer vision library"],
});

const methods = [
  ["Fotonet(model_path=None, nc=None, task='detect', device=None)", "Construct a canonical graph or load a supported native/exported artifact."],
  ["predict(source, imgsz=None, conf=0.25, max_det=300, batch=16, stream=False, retain_images=None)", "Predict images, folders, tensors, URLs, video, or a webcam."],
  ["predict_bgr(source, imgsz=None, conf=0.25, max_det=300, batch=16)", "Predict OpenCV-style BGR arrays without an RGB conversion at the call site."],
  ["track(source, persist=False, tracker='iou', tracker_iou=0.3, max_age=30, **kwargs)", "Attach same-class IoU track IDs to prediction results."],
  ["val(data, imgsz=None, batch=8, conf=0.0, max_det=100, ...)", "Evaluate an explicit validation source with ranked and operating-point metrics."],
  ["train(data, epochs=100, imgsz=640, batch=16, ...)", "Train or resume the uniform production protocol."],
  ["save(path, inference_only=False, half=False)", "Write a self-identifying native checkpoint."],
  ["export(path=None, format='onnx', imgsz=640, batch=1, dynamic=False, ...)", "Build and verify a versioned deployment artifact."],
] as const;

export default function ApiPage() {
  return (
    <>
      <DocHeader eyebrow="Reference / Python" title="The public surface, with return behavior attached." lead="The top-level package exposes one model facade plus result, box, and spatial-transform types. Internal graph classes are not the user API." />
      <section className="doc-section prose">
        <h2 id="imports">Public imports</h2>
        <CodeBlock code={`from fotonet import (
    Fotonet,
    Results,
    DetectionBox,
    DetectionBoxes,
    Vector2,
    AnchorPoint,
    FocusRegion,
    BoxTransform,
)`} label="Python" />
      </section>
      <section className="doc-section prose">
        <h2 id="model">Fotonet methods</h2>
        <dl className="api-list">{methods.map(([name, detail]) => <div key={name}><dt><code>{name}</code></dt><dd>{detail}</dd></div>)}</dl>
      </section>
      <section className="doc-section prose">
        <h2 id="returns">Prediction return contract</h2>
        <div className="table-wrap comparison-table">
          <table>
            <caption>Return type by input mode</caption>
            <thead><tr><th>Input</th><th>Return</th><th>Memory behavior</th></tr></thead>
            <tbody>
              <tr><th>One image</th><td><code>list[Results]</code> of length one</td><td>Source pixels retained</td></tr>
              <tr><th>Image list/folder/tensor batch</th><td><code>list[Results]</code></td><td>One item per source, same order</td></tr>
              <tr><th>Video/webcam, <code>stream=True</code></th><td><code>Iterator[Results]</code></td><td>Frames yielded incrementally</td></tr>
              <tr><th>Video/webcam, <code>stream=False</code></th><td><code>list[Results]</code></td><td>Results accumulated in memory</td></tr>
            </tbody>
          </table>
        </div>
        <Note title="Single-image indexing"><p>Use <code>result = model.predict(...)[0]</code>. Iterating <code>model.predict(...)</code> iterates images; iterating <code>result.boxes</code> iterates detections.</p></Note>
      </section>
      <section className="doc-section prose">
        <h2 id="results">Results and boxes</h2>
        <dl className="api-list">
          <div><dt><code>Results.boxes</code></dt><dd>Iterable <code>DetectionBoxes</code> collection.</dd></div>
          <div><dt><code>Results.plot() / show() / save()</code></dt><dd>Render detections when source pixels were retained.</dd></div>
          <div><dt><code>Results.to_json()</code></dt><dd>Serialize class, name, confidence, pixel <code>xyxy</code>, and optional track ID.</dd></div>
          <div><dt><code>Results.cpu() / to(device)</code></dt><dd>Return tensor-moved copies without mutating the original.</dd></div>
          <div><dt><code>DetectionBox.xywh / xyxy</code></dt><dd>Normalized coordinates for one detection.</dd></div>
          <div><dt><code>DetectionBox.transform</code></dt><dd>Mutable application-region helper for crop and geometry operations.</dd></div>
        </dl>
      </section>
      <ExampleRepositories topic="api" />
      <NextLinks items={[
        { href: "/docs/inference", label: "Inference guide", detail: "Inputs, batching, video, and practical prediction controls" },
        { href: "/docs/transforms", label: "Spatial API", detail: "Crops, anchors, movement, expansion, and containment" },
      ]} />
    </>
  );
}
