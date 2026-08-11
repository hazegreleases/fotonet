import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { ExampleRepositories } from "../../ui/ExampleRepositories";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "PyTorch Object Detection Checkpoints",
  description: "Understand fotonet full training checkpoints, slim inference weights, safe tensor-only loading, graph identities, class schemas, and resume compatibility.",
  path: "/docs/checkpoints",
  keywords: ["PyTorch checkpoint", "object detection weights", "resume training", "safe model loading"],
});

export default function CheckpointsPage() {
  return (
    <>
      <DocHeader eyebrow="Reference / checkpoints" title="A checkpoint’s contents decide what it can resume." lead="Native files identify their graph and class schema before tensors load. Full training checkpoints and slim inference checkpoints intentionally serve different jobs." />
      <section className="doc-section prose">
        <h2 id="types">Full and slim checkpoints</h2>
        <div className="table-wrap comparison-table">
          <table>
            <caption>Checkpoint capability matrix</caption>
            <thead><tr><th>Capability</th><th>Full last checkpoint</th><th>Slim inference checkpoint</th></tr></thead>
            <tbody>
              <tr><th>Prediction</th><td>Yes</td><td>Yes</td></tr>
              <tr><th>Export</th><td>Yes</td><td>Yes</td></tr>
              <tr><th>Weights-only initialization</th><td>Yes</td><td>Yes; training O2M heads are rebuilt compatibly</td></tr>
              <tr><th>Exact resume</th><td>Yes</td><td>No</td></tr>
              <tr><th>Optimizer, scheduler, scaler, RNG</th><td>Present</td><td>Removed</td></tr>
            </tbody>
          </table>
        </div>
      </section>
      <section className="doc-section prose">
        <h2 id="load">Load for inference</h2>
        <CodeBlock code={`from fotonet import Fotonet

model = Fotonet("weights/fotonetn.pt")
results = model.predict("image.jpg", conf=0.25)`} label="Python" />
        <p>The loader cross-checks checkpoint format, architecture schema, canonical model ID, architecture fingerprint, P2 setting, class count, and ordered class names. Missing or conflicting identity data fails closed.</p>
        <Note title="Tensor-only native loading"><p>There is no public unsafe-pickle option. ONNX and TorchScript are executable model programs and should still be treated as trusted artifacts.</p></Note>
      </section>
      <section className="doc-section prose">
        <h2 id="resume">Resume versus pretrained</h2>
        <CodeBlock code={`model = Fotonet("runs/fotonetn/fotonet_last.pt")
summary = model.train(
    data="data.yaml",
    epochs=300,
    weights="runs/fotonetn/fotonet_last.pt",
    resume=True,
    save_dir="runs/fotonetn",
)`} label="Python / exact continuation" />
        <CodeBlock code={`model = Fotonet("fotonetn", nc=2)
summary = model.train(
    data="data.yaml",
    epochs=300,
    weights="weights/source.pt",
    pretrained=True,
    save_dir="runs/fresh-run",
)`} label="Python / fresh training state" />
        <p><code>resume=True</code> and <code>pretrained=True</code> are mutually exclusive. Resume restores the same run; pretrained starts a new optimizer and schedule from compatible tensors.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="save">Save an inference checkpoint</h2>
        <CodeBlock code={`model.save(
    "weights/fotonetn-inference.pt",
    inference_only=True,
    half=False,
)`} label="Python" />
        <p>Use full precision unless the target has been checked for FP16. Release weights belong on GitHub Releases with an immutable checksum, not in Git history.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="official-weight">Official weight status</h2>
        <p>The first official Nano weight is currently training. After training and release verification, the project will publish the weight, byte size, SHA256 checksum, canonical COCO report, and an automatic model-name download hook.</p>
      </section>
      <ExampleRepositories topic="checkpoints" />
      <NextLinks items={[
        { href: "/docs/training", label: "Training and resume", detail: "Use the strict launcher for interruption recovery" },
        { href: "/docs/export", label: "Export a checkpoint", detail: "Carry preprocessing and output metadata with the graph" },
      ]} />
    </>
  );
}
