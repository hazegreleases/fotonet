import { modelGuidance, modelMetrics } from "../data";
import { pageMetadata } from "../seo";
import { AmbientShapes } from "../ui/AmbientShapes";

export const metadata = pageMetadata({
  title: "Object Detection Model Benchmarks",
  description: "Measured fotonet parameter counts, MACs, GFLOPs, latency, throughput, and GPU memory for ten compact object detection models at 640 by 640.",
  path: "/benchmarks",
  keywords: ["object detection benchmark", "inference latency", "GFLOPs", "model parameters", "RTX 4060"],
});

function MeasurementTable({ p2 }: { p2: boolean }) {
  const rows = modelMetrics.filter((row) => row.p2 === p2);
  return (
    <div className="table-wrap measurement-table">
      <table>
        <caption>{p2 ? "P2–P5 small-object variants" : "Standard P3–P5 variants"}</caption>
        <thead>
          <tr>
            <th>Model</th>
            <th>Parameters<br /><small>full → deploy</small></th>
            <th>Compute<br /><small>GMAC / GFLOPs</small></th>
            <th>Single image<br /><small>p50 ms / FPS</small></th>
            <th>Batch 8<br /><small>images/s</small></th>
            <th>Peak memory<br /><small>B1 / B8 MiB</small></th>
            <th>Candidates<br /><small>at 640²</small></th>
          </tr>
        </thead>
        <tbody>{rows.map((row) => (
          <tr key={row.name}>
            <th><code>{row.name}</code></th>
            <td>{row.fullParams} <span className="comparison-arrow">→</span> {row.deployParams}</td>
            <td>{row.gmac} / {row.gflops}</td>
            <td>{row.latency1} / {row.fps1}</td>
            <td>{row.fps8}</td>
            <td>{row.vram1} / {row.vram8}</td>
            <td>{row.candidates}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export default function BenchmarksPage() {
  return (
    <main id="content" className="benchmark-page">
      <header className="benchmark-header">
        <AmbientShapes variant="technical" />
        <p className="eyebrow">Measurement record / 2026-08-10</p>
        <h1>Deployment cost at 640 × 640.</h1>
        <p>All ten stable graphs were measured without training or dataset access. These numbers describe raw native inference, not accuracy or application throughput.</p>
      </header>
      <section className="benchmark-spec">
        <dl>
          <div><dt>GPU</dt><dd>GeForce RTX 4060 · 8 GB</dd></div>
          <div><dt>Runtime</dt><dd>PyTorch 2.11 · CUDA 12.8 · cuDNN 9.19</dd></div>
          <div><dt>Graph</dt><dd>Fused eager FP32 · TF32 off</dd></div>
          <div><dt>Input</dt><dd>Zero-filled CUDA tensor · 640 × 640</dd></div>
          <div><dt>Timing</dt><dd>30 warmups · 100 synchronized forwards</dd></div>
          <div><dt>Compute</dt><dd>THOP 0.1.1 · FLOPs = 2 × MACs</dd></div>
        </dl>
      </section>
      <section className="metric-summary" aria-label="Benchmark highlights">
        <article><span>Smallest deploy graph</span><strong>1.006M</strong><p><code>fotonetn</code> parameters after stripping O2M and fusion.</p></article>
        <article><span>Fastest single image</span><strong>4.225 ms</strong><p><code>fotonetn</code> median raw forward at batch 1.</p></article>
        <article><span>Highest batch throughput</span><strong>723 img/s</strong><p><code>fotonetn</code> arithmetic-mean throughput at batch 8.</p></article>
        <article><span>P2 output expansion</span><strong>4.05×</strong><p>8,400 to 34,000 candidates before confidence filtering.</p></article>
      </section>
      <section className="pair-comparison-section">
        <div className="benchmark-section-heading">
          <p className="eyebrow">The useful comparison</p>
          <h2>Standard models compared with P2 variants.</h2>
          <p>These deltas compare each capacity profile with its own P2 sibling. Negative throughput is a cost, not a quality verdict.</p>
        </div>
        <div className="table-wrap comparison-table">
          <table>
            <caption>P2 cost relative to the matching standard model</caption>
            <thead><tr><th>Profile</th><th>Model pair</th><th>Compute increase</th><th>Batch-8 throughput</th><th>Peak B8 memory</th><th>Candidate count</th></tr></thead>
            <tbody>{modelGuidance.map((pair) => {
              const standard = modelMetrics.find((item) => item.name === pair.standard)!;
              const p2 = modelMetrics.find((item) => item.name === pair.p2)!;
              const compute = ((Number(p2.gmac) / Number(standard.gmac) - 1) * 100).toFixed(0);
              const throughput = ((Number(p2.fps8) / Number(standard.fps8) - 1) * 100).toFixed(0);
              const memory = ((Number(p2.vram8) / Number(standard.vram8) - 1) * 100).toFixed(0);
              return (
                <tr key={pair.profile}>
                  <th><span className="profile-badge">{pair.profile}</span></th>
                  <td><code>{pair.standard}</code><br /><small>vs {pair.p2}</small></td>
                  <td><span className="delta delta-cost">+{compute}%</span></td>
                  <td>{standard.fps8} → {p2.fps8}<br /><span className="delta delta-cost">{throughput}%</span></td>
                  <td>{standard.vram8} → {p2.vram8} MiB<br /><span className="delta delta-cost">+{memory}%</span></td>
                  <td>8,400 → 34,000<br /><span className="delta delta-info">4.05×</span></td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
      </section>
      <section className="benchmark-table-section">
        <div className="benchmark-section-heading">
          <p className="eyebrow">Complete measurement record</p>
          <h2>Standard and P2 model benchmarks.</h2>
          <p>Each cell keeps paired quantities together: full/deploy parameters, MACs/FLOPs, median latency/FPS, and batch-1/batch-8 memory.</p>
        </div>
        <MeasurementTable p2={false} />
        <MeasurementTable p2 />
      </section>
      <section className="benchmark-notes">
        <div><span>01</span><h2>Full versus deploy parameters</h2><p>Full counts include the training-only O2M branches and BatchNorm affine parameters. Deploy counts follow <code>eval() → strip_o2m_for_inference() → fuse()</code>. Every fused graph contains zero BatchNorm modules.</p></div>
        <div><span>02</span><h2>FPS convention</h2><p>FPS is batch size divided by arithmetic mean latency. The displayed B1 latency is p50, so it is not the reciprocal used for FPS. Windows scheduling noise is visible in p95 tails.</p></div>
        <div><span>03</span><h2>What THOP leaves out</h2><p>MAC totals cover the complete fused graph but do not charge unsupported elementwise, concatenation, indexing, or grid-construction operations.</p></div>
        <div><span>04</span><h2>What remains unknown</h2><p>No number here establishes COCO AP, robustness, small-object recall, or parity with an older graph. Those require trained checkpoints and controlled evaluation.</p></div>
      </section>
    </main>
  );
}
