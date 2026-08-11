import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { ExampleRepositories } from "../../ui/ExampleRepositories";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "Python Object Detection Inference",
  description: "Run NMS-free object detection on images, folders, tensors, video, or webcams and work with typed fotonet Results and detection boxes in Python.",
  path: "/docs/inference",
  keywords: ["object detection inference", "Python computer vision", "image detection", "video detection"],
});

export default function InferencePage() {
  return (
    <>
      <DocHeader eyebrow="How-to / inference" title="Predict without losing the source-to-result mapping." lead="Image and tensor calls return one Results object per source. Only video and webcam calls with stream=True return an iterator." />
      <section className="doc-section prose">
        <h2 id="images">Images and folders</h2>
        <CodeBlock code={`from fotonet import Fotonet\n\nmodel = Fotonet("path/to/checkpoint.pt")\nresults = model.predict("images", batch=8, conf=0.25, imgsz=640)\n\nfor result in results:\n    print(result)`} label="Python" />
        <p><code>len(results)</code> is the number of inputs. <code>len(result)</code> is the number of detections in that input.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="bgr">OpenCV BGR frames</h2>
        <CodeBlock code={`result = model.predict_bgr(frame, conf=0.25)[0]`} label="Python / OpenCV" />
      </section>
      <section className="doc-section prose">
        <h2 id="tensors">Tensor inputs</h2>
        <p>Tensor input accepts RGB <code>[3,H,W]</code> or <code>[B,3,H,W]</code> values in <code>[0,1]</code> or <code>[0,255]</code>. Tensor inference avoids retaining a batch-sized CPU copy by default.</p>
        <CodeBlock code={`results = model.predict(images, retain_images=True)`} label="Python / retain source pixels" />
        <p>Set <code>retain_images=True</code> only when you need <code>plot()</code>, <code>save()</code>, or crop helpers.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="results">Results reference</h2>
        <dl className="definition-list">
          <div><dt><code>orig_img</code></dt><dd>The retained source image or array.</dd></div>
          <div><dt><code>boxes</code></dt><dd>Iterable collection of <code>DetectionBox</code> objects.</dd></div>
          <div><dt><code>scores</code></dt><dd>Confidence tensor.</dd></div>
          <div><dt><code>classes</code></dt><dd>Class-id tensor.</dd></div>
          <div><dt><code>names</code></dt><dd>Ordered class-name mapping saved with the checkpoint.</dd></div>
        </dl>
        <CodeBlock code={`plot = result.plot()\nresult.save("exports/prediction.jpg")\njson_text = result.to_json()`} label="Python / output helpers" />
      </section>
      <section className="doc-section prose">
        <h2 id="tracking">Dependency-free tracking</h2>
        <CodeBlock code={`tracked = model.track("video.mp4", stream=True, persist=True)\nfor result in tracked:\n    print(result.to_json())`} label="Python" />
        <Note title="Tracker scope"><p><code>tracker=&quot;iou&quot;</code> is IoU association. It does not claim motion prediction or re-identification.</p></Note>
      </section>
      <section className="doc-section prose">
        <h2 id="static-artifacts">Static exported artifacts</h2>
        <p>A static ONNX or TorchScript artifact defaults to its recorded input size and rejects an incompatible explicit <code>imgsz</code>. Export with <code>dynamic=True</code> when variable shapes are required.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="predict-options">Prediction controls</h2>
        <dl className="api-list">
          <div><dt><code>conf=0.25</code></dt><dd>Discard candidates below the confidence threshold after class selection.</dd></div>
          <div><dt><code>max_det=300</code></dt><dd>Cap detections per image after ranking.</dd></div>
          <div><dt><code>batch=1</code></dt><dd>Batch expanded image sources while preserving one result per source.</dd></div>
          <div><dt><code>retain_images</code></dt><dd>Keep source pixels for plotting, saving, and crops; tensor inputs default to not retaining them.</dd></div>
          <div><dt><code>stream=True</code></dt><dd>Yield video or webcam results without accumulating the entire stream.</dd></div>
        </dl>
        <Note title="NMS-free means no hidden NMS switch"><p>The production runtime performs score selection, confidence filtering, clipping, and top-k limiting. It does not expose a second NMS inference path.</p></Note>
      </section>
      <ExampleRepositories topic="inference" />
      <NextLinks items={[
        { href: "/docs/transforms", label: "Manipulate detections", detail: "Coordinates, anchors, crops, and containment" },
        { href: "/docs/export", label: "Export the model", detail: "Tensor contract, metadata, and backend support" },
      ]} />
    </>
  );
}
