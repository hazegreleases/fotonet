import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { ExampleRepositories } from "../../ui/ExampleRepositories";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "Export Object Detection to ONNX and TorchScript",
  description: "Export fotonet object detection models to ONNX, TorchScript, TensorRT, or CoreML while preserving tensor shapes, class schemas, and runtime metadata.",
  path: "/docs/export",
  keywords: ["ONNX object detection", "TorchScript export", "TensorRT detector", "CoreML computer vision"],
});

export default function ExportPage() {
  return (
    <>
      <DocHeader eyebrow="How-to / deployment" title="Export the graph and its coordinate contract together." lead="ONNX sidecars and TorchScript embedded metadata record class names, RGB preprocessing, input shape, raw output layout, and stride-padding behavior." />
      <section className="doc-section prose">
        <h2 id="onnx">ONNX</h2>
        <CodeBlock code={`from fotonet import Fotonet\n\nmodel = Fotonet("path/to/checkpoint.pt")\nartifact = model.export(\n    format="onnx",\n    path="exports/fotonet.onnx",\n    imgsz=640,\n)\nprint(artifact["artifact"], artifact["metadata"])`} label="Python" />
        <p>By default, export checks native versus ONNX Runtime numerical parity. Static artifacts accept only their recorded input shape.</p>
        <CodeBlock code={`model.export(\n    format="onnx",\n    path="exports/fotonet_dynamic.onnx",\n    imgsz=(640, 960),\n    dynamic=True,\n)`} label="Python / dynamic H and W" />
      </section>
      <section className="doc-section prose">
        <h2 id="torchscript">TorchScript</h2>
        <CodeBlock code={`model.export(\n    format="torchscript",\n    path="fotonet.torchscript",\n    imgsz=640,\n)`} label="Python" />
        <p>TorchScript embeds FOTO-NET metadata, so moving the archive alone preserves its class and preprocessing contract. Set <code>dynamic=True</code> to validate variable batch and spatial dimensions.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="accelerators">TensorRT and CoreML</h2>
        <CodeBlock code={`model.export(\n    format="tensorrt",\n    path="fotonet.engine",\n    imgsz=640,\n    half=True,\n)`} label="Python / TensorRT" />
        <p>TensorRT requires <code>trtexec</code> in <code>PATH</code>. CoreML support exists, but it is not considered certified without fresh platform-specific verification.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="output">Raw output</h2>
        <div className="tensor-contract"><span>B</span><span>N</span><span>nc class logits</span><span>4 × normalized xywh</span></div>
        <p>The raw shape is <code>[B, N, nc + 4]</code>. The runtime adapter applies sigmoid, class selection, confidence filtering, clipping, and a top-k cap for ONNX and TorchScript prediction. The production path remains NMS-free.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="int8">INT8 ONNX</h2>
        <CodeBlock code={`artifact = model.export(\n    format="onnx",\n    path="exports/fotonet_int8.onnx",\n    imgsz=640,\n    int8=True,\n    calibration_data=calibration_batches,\n)`} label="Python" />
        <Note title="Calibration is part of the build input" tone="warning"><p>Supply representative RGB NCHW batches in <code>[0,1]</code>. Treat calibration data, artifacts, metadata, and export toolchains as trusted inputs.</p></Note>
      </section>
      <ExampleRepositories topic="export" />
      <NextLinks items={[
        { href: "/docs/inference", label: "Load an exported artifact", detail: "Static shape and Results behavior" },
        { href: "/benchmarks", label: "Understand published speed", detail: "Current numbers are native eager FP32" },
      ]} />
    </>
  );
}
