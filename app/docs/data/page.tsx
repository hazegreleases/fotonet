import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { ExampleRepositories } from "../../ui/ExampleRepositories";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "Object Detection Datasets and YOLO Labels",
  description: "Configure YOLO-format object detection datasets, ordered class schemas, annotation validation, image discovery, and training data policies for fotonet.",
  path: "/docs/data",
  keywords: ["YOLO labels", "object detection dataset", "bounding box annotations", "COCO dataset"],
});

export default function DataPage() {
  return (
    <>
      <DocHeader eyebrow="How-to / data" title="Make the dataset contract explicit before training." lead="A dataset is paths, normalized detection labels, and one ordered class schema shared by training, validation, checkpoints, and results." />
      <section className="doc-section prose">
        <h2 id="layout">Recommended directory layout</h2>
        <CodeBlock code={`datasets/animals/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/`} language="text" label="Directory tree" />
        <p>Each image label uses the same relative stem: <code>images/train/frame_001.jpg</code> maps to <code>labels/train/frame_001.txt</code>.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="yaml">Dataset YAML</h2>
        <CodeBlock code={`path: datasets/animals
train: images/train
val: images/val
nc: 2
names:
  0: cat
  1: dog`} language="yaml" label="data.yaml" />
        <p><code>path</code> is the root used to resolve relative train and validation sources. <code>nc</code> must equal the number of names, and numeric name keys must be contiguous from zero.</p>
        <Note title="Class order is semantic identity" tone="warning"><p>Changing <code>0: cat, 1: dog</code> to <code>0: dog, 1: cat</code> is not compatible even though <code>nc</code> remains two. Resume and checkpoint loading compare ordered names.</p></Note>
      </section>
      <section className="doc-section prose">
        <h2 id="labels">YOLO detection labels</h2>
        <CodeBlock code={`# class_id x_center y_center width height
0 0.512500 0.481250 0.300000 0.425000
1 0.204688 0.718750 0.128125 0.212500`} language="text" label="labels/train/frame_001.txt" />
        <p>Coordinates are normalized to the source image. Empty label files represent valid background images. A missing label is rejected unless <code>allow_missing_labels=True</code> is chosen deliberately.</p>
      </section>
      <section className="doc-section prose">
        <h2 id="audit">Annotation policy</h2>
        <dl className="definition-list">
          <div><dt><code>annotation_policy=&quot;fix&quot;</code></dt><dd>Clamp repairable edge crossings and drop malformed or duplicate rows while recording the audit.</dd></div>
          <div><dt><code>annotation_policy=&quot;error&quot;</code></dt><dd>Fail immediately when annotations violate the accepted contract.</dd></div>
          <div><dt><code>source_recursive=True</code></dt><dd>Discover supported images recursively under a directory source.</dd></div>
          <div><dt><code>cache_labels=True</code></dt><dd>Cache derived label parsing; original images and annotations remain authoritative.</dd></div>
        </dl>
        <p>Public validation requires an explicit <code>val</code> source. It never substitutes training data when validation data is absent.</p>
      </section>
      <ExampleRepositories topic="data" />
      <NextLinks items={[
        { href: "/docs/training", label: "Start training", detail: "Dry-run the graph, data, and run identity first" },
        { href: "/docs/validation", label: "Validate the dataset", detail: "COCO ranking and annotation-policy reporting" },
      ]} />
    </>
  );
}
