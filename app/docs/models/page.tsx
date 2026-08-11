import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { CodeBlock } from "../../ui/CodeBlock";
import { modelGuidance, modelMetrics } from "../../data";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "Compact Object Detection Models",
  description: "Compare fotonet N, S, M, L, and X object detector profiles, small-object P2 variants, feature strides, parameter counts, and graph identity rules.",
  path: "/docs/models",
  keywords: ["compact object detector", "small object detection", "P2 detector", "model scaling"],
});

export default function ModelsPage() {
  return (
    <>
      <DocHeader eyebrow="Reference / graph" title="Ten canonical names, one production model family." lead="Every public scale uses the same production V1 Backbone, Neck, Head, and Detector contract. Named models are architecture definitions; the first official Nano weight is still training." />
      <section className="doc-section prose">
        <h2 id="matrix">Choose the profile first, then decide on P2</h2>
        <p>The letter selects foundation capacity. The <code>-p2</code> suffix is a separate small-object trade: it adds a stride-4 output, more candidates, and substantially more activation traffic.</p>
        <div className="table-wrap comparison-table">
          <table>
            <caption>Stable names grouped by capacity profile</caption>
            <thead><tr><th>Profile</th><th>Typical role</th><th>Standard P3–P5</th><th>Optional P2–P5</th><th>Graph</th><th>Published AP</th></tr></thead>
            <tbody>{modelGuidance.map((row) => (
              <tr key={row.profile}><th><span className="profile-badge">{row.profile}</span></th><td>{row.role}</td><td><code>{row.standard}</code></td><td><code>{row.p2}</code></td><td>Production V1</td><td><span className="unknown-value">Pending weight</span></td></tr>
            ))}</tbody>
          </table>
        </div>
      </section>
      <section className="doc-section prose model-cost-section">
        <h2 id="cost">Measured cost of each pair</h2>
        <p>Every row compares siblings on the same RTX 4060 FP32 protocol. Throughput is raw batch-8 model throughput—not camera or application FPS.</p>
        <div className="table-wrap comparison-table">
          <table>
            <caption>Standard model compared directly with its P2 sibling at 640 × 640</caption>
            <thead><tr><th>Profile</th><th>Deploy params<br /><small>standard → P2</small></th><th>GFLOPs<br /><small>standard → P2</small></th><th>Batch-8 img/s<br /><small>standard → P2</small></th><th>Peak memory<br /><small>standard → P2</small></th></tr></thead>
            <tbody>{modelGuidance.map((pair) => {
              const standard = modelMetrics.find((item) => item.name === pair.standard)!;
              const p2 = modelMetrics.find((item) => item.name === pair.p2)!;
              return (
                <tr key={pair.profile}>
                  <th><span className="profile-badge">{pair.profile}</span></th>
                  <td>{standard.deployParams} <span className="comparison-arrow">→</span> {p2.deployParams}</td>
                  <td>{standard.gflops} <span className="comparison-arrow">→</span> {p2.gflops}</td>
                  <td>{standard.fps8} <span className="comparison-arrow cost-down">→</span> {p2.fps8}</td>
                  <td>{standard.vram8} <span className="comparison-arrow cost-up">→</span> {p2.vram8} MiB</td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
      </section>
      <section className="doc-section prose">
        <h2 id="p2">When P2 is worth its cost</h2>
        <p>P2 variants add a stride-4 feature level to the neck and head. They preserve the same backbone and foundation fingerprint as their non-P2 sibling, but increase raw candidate count from 8,400 to 34,000 at 640×640.</p>
        <Note title="P2 is targeted, not automatically better" tone="warning"><p>Use P2 when small-object validation supports it. On the measured RTX 4060, P2’s activation and candidate traffic raised latency and memory materially even when parameter growth was small.</p></Note>
      </section>
      <section className="doc-section prose">
        <h2 id="config">Configuration contract</h2>
        <CodeBlock code={`nc: 80\nprofile: n\np2: false\nreg_max: 12\nquality_head: false\narchitecture_schema: 1`} language="yaml" label="fotonetn.yaml" />
        <p>The profile resolves reviewed integer channels, depths, downsampling modes, and neck widths. Arbitrary YAML layer lists, aliases, and undeclared graph mutations are rejected.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="identity">Checkpoint identity</h2>
        <dl className="definition-list">
          <div><dt><code>checkpoint_format</code></dt><dd>Version of the native checkpoint container.</dd></div>
          <div><dt><code>architecture_schema</code></dt><dd>Version of the production graph configuration contract.</dd></div>
          <div><dt><code>model_id</code></dt><dd>One of the ten canonical N/S/M/L/X or P2 names.</dd></div>
          <div><dt><code>architecture_fingerprint</code></dt><dd>Deterministic SHA-256 over the complete resolved graph specification and class count.</dd></div>
        </dl>
        <p>Arbitrary YAML layer lists are rejected. A configuration name, checkpoint identity, and instantiated graph must agree.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="future-scales">Planned scale rebalance</h2>
        <p>S, M, L, and X will be resized in a later architecture revision. These are design bands, not current measurements or accuracy forecasts.</p>
        <div className="table-wrap comparison-table">
          <table>
            <caption>Future parameter targets</caption>
            <thead><tr><th>Scale</th><th>Target center</th><th>Allowed range</th></tr></thead>
            <tbody>
              <tr><th>S</th><td>2.20M</td><td>2.02M–2.38M</td></tr>
              <tr><th>M</th><td>5.00M</td><td>4.50M–5.50M</td></tr>
              <tr><th>L</th><td>11.40M</td><td>10.40M–12.40M</td></tr>
              <tr><th>X</th><td>33.80M</td><td>28.80M–38.80M</td></tr>
            </tbody>
          </table>
        </div>
        <p>Future MACs, FLOPs, memory, and latency will be measured only after those graphs exist.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="foundation">Foundation outline</h2>
        <ol className="numbered-steps">
          <li><b>Backbone</b><span>Spatial downsampling and CSP refinement produce P2–P5 backbone features; SPPF expands the deepest receptive field.</span></li>
          <li><b>Neck</b><span>Concat FPN/PAN fusion returns P3–P5, or P2–P5 for a P2 variant.</span></li>
          <li><b>Head</b><span>One-to-one output is used for NMS-free inference; one-to-many branches exist only for training supervision.</span></li>
        </ol>
      </section>
      <NextLinks items={[
        { href: "/benchmarks", label: "Compare measured cost", detail: "All models at one declared runtime" },
        { href: "/docs/training", label: "Train the selected graph", detail: "Strict launch and resume contract" },
      ]} />
    </>
  );
}
