import type { Metadata } from "next";
import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { inferenceCode, installCode } from "../../data";

export const metadata: Metadata = { title: "Install and first run" };

export default function InstallPage() {
  return (
    <>
      <DocHeader eyebrow="Tutorial / start" title="Install FOTO-NET and run one checkpoint." lead="The package targets Python 3.10 or newer. Install the PyTorch build appropriate for your platform first when CUDA support matters." />
      <section className="doc-section prose">
        <h2 id="install">1. Install the package</h2>
        <CodeBlock code={installCode} language="bash" label="Terminal" />
        <p>For a development checkout:</p>
        <CodeBlock code={`git clone https://github.com/hazegreleases/fotonet.git\ncd fotonet\npython -m pip install -e ".[dev]"`} language="bash" label="Terminal / source checkout" />
      </section>
      <section className="doc-section prose">
        <h2 id="checkpoint">2. Bring a trusted checkpoint</h2>
        <Note title="A model name is not a pretrained model" tone="warning">
          <p><code>Fotonet(&quot;fotonetn&quot;)</code> constructs an untrained architecture. The first official Nano weight is training and will later arrive through a checksummed GitHub Release and automatic download hook.</p>
        </Note>
        <p>Until that release, use a checkpoint you trained or independently trust. Native checkpoints load in tensor-only mode and must carry an explicit, supported schema and graph identity. There is no unsafe-pickle override or metadata-free graph guessing.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="predict">3. Predict one image</h2>
        <CodeBlock code={inferenceCode} label="Python" />
        <p><code>predict()</code> returns a list even for one image. Choose <code>results[0]</code> before iterating its boxes.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="cli">CLI equivalent</h2>
        <CodeBlock code={`fotonet predict model=my_checkpoint.pt source=image.jpg conf=0.25 save=true`} language="bash" label="Terminal" />
      </section>
      <NextLinks items={[
        { href: "/docs/inference", label: "Continue to inference", detail: "Folders, tensors, BGR, video, and Results" },
        { href: "/docs/checkpoints", label: "Understand checkpoints", detail: "Identity, full resume state, and slim inference files" },
      ]} />
    </>
  );
}
