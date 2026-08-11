import Link from "next/link";
import { CodeBlock } from "./ui/CodeBlock";
import { AmbientShapes } from "./ui/AmbientShapes";
import { inferenceCode, modelMetrics } from "./data";

export default function Home() {
  return (
    <main id="content">
      <section className="hero ruled-section">
        <AmbientShapes variant="hero" />
        <div className="hero-copy">
          <p className="eyebrow">Production V1 / one graph contract</p>
          <h1>fotonet<br /><em>&quot;yet another alternative&quot;</em></h1>
          <p className="hero-lead">
            FOTO-NET is a compact, NMS-free object detector with a Python API,
            multi-scale model family, resumable training, and deployment exports.
            The first official Nano weight is training now. Graph cost is measured;
            accuracy will be published only with the finished weight and canonical evaluation.
          </p>
          <div className="hero-actions">
            <Link className="button-primary" href="/docs/install">Start with the manual <span aria-hidden="true">→</span></Link>
            <Link className="text-link" href="/benchmarks">Read measured performance</Link>
          </div>
        </div>
        <aside className="fact-sheet" aria-label="Current project facts">
          <div className="fact-sheet-heading"><span>Current state</span><span>2026—08—11</span></div>
          <dl>
            <div><dt>Stable scales</dt><dd>10</dd></div>
            <div><dt>Nano deploy params</dt><dd>1.006M</dd></div>
            <div><dt>Nano raw FP32</dt><dd>218 FPS*</dd></div>
            <div><dt>Raw output</dt><dd>[B, N, nc+4]</dd></div>
            <div><dt>License</dt><dd>Apache-2.0</dd></div>
          </dl>
          <p>* RTX 4060, 640², batch 1, fused eager graph. Not end-to-end.</p>
        </aside>
      </section>

      <section className="status-strip" aria-label="Release status">
        <strong>Beta boundary:</strong>
        <span>architectures and full workflows exist</span>
        <span>official Nano weight is currently training</span>
        <span>no COCO AP claim has been published</span>
      </section>

      <section className="split-section ruled-section">
        <div className="section-index">01 / USE</div>
        <div className="section-heading">
          <p className="eyebrow">Inference starts from a checkpoint</p>
          <h2>And here is how to run inference.</h2>
          <p>
            Image prediction returns one <code>Results</code> object per input.
            Each result owns its source image, boxes, scores, class mapping, plotting,
            JSON serialization, and spatial transforms.
          </p>
          <Link className="text-link" href="/docs/inference">Inference guide →</Link>
        </div>
        <CodeBlock code={inferenceCode} label="Python / inference" />
      </section>

      <section className="architecture ruled-section">
        <AmbientShapes variant="technical" />
        <div className="section-index">02 / GRAPH</div>
        <div className="architecture-copy">
          <p className="eyebrow">One explicit foundation</p>
          <h2>Production graph overview.</h2>
          <p>
            Stable N/S/M/L/X names resolve to reviewed integer channel and depth settings.
            P2 variants extend the neck and head for small objects without changing the backbone.
          </p>
        </div>
        <ol className="graph-flow" aria-label="FOTO-NET graph overview">
          <li className="graph-backbone"><b>01</b><span>Backbone</span><small>CSP refinement · spatial downsampling · SPPF</small><em>P2—P5</em></li>
          <li className="graph-neck"><b>02</b><span>Neck</span><small>Concat FPN/PAN · three or four feature scales</small><em>FUSE</em></li>
          <li className="graph-head"><b>03</b><span>Head</span><small>O2O inference · training-only O2M supervision</small><em>O2O</em></li>
          <li className="graph-results"><b>04</b><span>Results</span><small>Normalized xywh · class scores · transform layer</small><em>API</em></li>
        </ol>
      </section>

      <section className="metrics-preview ruled-section">
        <div className="section-index">03 / MEASURE</div>
        <div className="metrics-intro">
          <div>
            <p className="eyebrow">No accuracy implied</p>
            <h2>Measured inference performance at 640 × 640.</h2>
          </div>
          <p>
            These are raw fused FP32 forwards at 640×640 on one RTX 4060.
            They exclude decoding, image I/O, preprocessing, and postprocessing.
          </p>
        </div>
        <div className="table-wrap">
          <table>
            <caption>Standard P3–P5 models on the same 640² FP32 protocol</caption>
            <thead><tr><th>Profile</th><th>Model</th><th>Deploy params</th><th>Compute<br /><small>GMAC</small></th><th>Single image<br /><small>FPS</small></th><th>Batch 8<br /><small>img/s</small></th><th>Peak B8<br /><small>MiB</small></th></tr></thead>
            <tbody>
              {modelMetrics.filter((row) => !row.p2).map((row) => (
                <tr key={row.name}>
                  <th><span className="profile-badge">{row.name.replace("fotonet", "").toUpperCase()}</span></th><td><code>{row.name}</code></td><td>{row.deployParams}</td><td>{row.gmac}</td><td>{row.fps1}</td><td>{row.fps8}</td><td>{row.vram8}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Link className="button-secondary" href="/benchmarks">All ten models and methodology →</Link>
      </section>

      <section className="manual-map ruled-section">
        <div className="section-index">04 / MANUAL</div>
        <div className="manual-intro">
          <p className="eyebrow">Documentation by task</p>
          <h2>Documentation by task.</h2>
        </div>
        <div className="manual-grid">
          <Link href="/docs/install"><span>Start</span><h3>Install and run a trusted checkpoint</h3><p>Requirements, installation, first inference, and the current release boundary.</p></Link>
          <Link href="/docs/training"><span>Workflow</span><h3>Train and resume without guessing</h3><p>Dataset schema, strict launcher, full checkpoints, and interruption recovery.</p></Link>
          <Link href="/docs/checkpoints"><span>Contract</span><h3>Know which checkpoint can do what</h3><p>Full resume state, slim inference weights, identities, and safe loading.</p></Link>
          <Link href="/docs/api"><span>API</span><h3>Look up the Python surface</h3><p>Constructor, prediction, validation, export, results, and return types.</p></Link>
          <Link href="/docs/models"><span>Reference</span><h3>Choose a scale or a P2 variant</h3><p>Graph identity, configuration rules, parameters, MACs, and trade-offs.</p></Link>
          <Link href="/docs/export"><span>Deployment</span><h3>Export with a declared tensor contract</h3><p>ONNX, TorchScript, TensorRT, CoreML, metadata, and dynamic shapes.</p></Link>
          <Link href="/docs/examples"><span>Examples</span><h3>Download scripts that run as written</h3><p>Image prediction, folder-to-JSONL inference, ONNX export, and training commands.</p></Link>
          <Link href="/benchmarks"><span>Evidence</span><h3>Read the complete measurement record</h3><p>Paired P2 comparisons, parameters, compute, latency, memory, and methodology.</p></Link>
        </div>
      </section>

      <section className="honesty ruled-section">
        <AmbientShapes variant="organic" />
        <p className="eyebrow">What this project does not claim</p>
        <h2>Current measurement and evaluation status.</h2>
        <div>
          <p>
            Parameter count, MACs, and latency describe the implementation. They do not
            establish COCO AP, rare-class recall, small-object quality, or parity with an older model.
          </p>
          <p>
            A public accuracy statement requires a released checkpoint, checksum,
            declared dataset split, exact evaluation protocol, and reproducible command.
          </p>
        </div>
      </section>
    </main>
  );
}
