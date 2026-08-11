import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks } from "../../ui/Docs";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "Detection Boxes, Results, and Image Transforms",
  description: "Use fotonet detection Results, normalized and pixel bounding boxes, anchors, crops, expansion, containment, and coordinate-safe image transforms.",
  path: "/docs/transforms",
  keywords: ["bounding box API", "image crop", "detection results", "coordinate transforms"],
});

const methods = [
  ["set_anchor(anchor)", "Choose the fixed point used by scale and expansion operations."],
  ["move((dx, dy))", "Move in normalized coordinates."],
  ["pixel_move((dx, dy))", "Move in source-image pixels."],
  ["scale((sx, sy))", "Scale about the active anchor."],
  ["expand(padding)", "Expand in normalized units."],
  ["pixel_expand(padding)", "Expand in pixels."],
  ["clamp()", "Clamp the region to the image bounds."],
  ["set_box(mode=1)", "Make the region square."],
  ["set_aspect_ratio((w, h), mode=1)", "Enforce an aspect ratio."],
  ["crop(image)", "Crop a PIL image or NumPy array."],
  ["contains(point=...)", "Test normalized point or region containment."],
  ["pixel_contains(point=...)", "Test containment in pixel coordinates."],
  ["iou(other)", "Compute intersection over union."],
  ["distance(point)", "Measure normalized distance from the active anchor."],
  ["focus(x, y, w, h)", "Focus on a relative subregion."],
  ["focus_reset()", "Restore the original detection box."],
];

export default function TransformsPage() {
  return (
    <>
      <DocHeader eyebrow="Reference / spatial API" title="Move from a detection to an application region." lead="Every DetectionBox exposes raw coordinates and a mutable transform helper for crops, movement, scaling, anchoring, containment, and relative focus." />
      <section className="doc-section prose">
        <h2 id="example">Crop around a detection</h2>
        <CodeBlock code={`from fotonet import AnchorPoint\n\nresult = results[0]\nbox = result.boxes[0]\ncrop = (\n    box.transform\n    .set_anchor(AnchorPoint.CENTER)\n    .pixel_expand(40)\n    .clamp()\n    .crop(result.orig_img)\n)`} label="Python" />
      </section>
      <section className="doc-section prose">
        <h2 id="coordinates">Coordinates and properties</h2>
        <dl className="definition-list">
          <div><dt><code>box.xywh</code></dt><dd>Original normalized center x, center y, width, and height.</dd></div>
          <div><dt><code>box.xyxy</code></dt><dd>Original normalized corner coordinates.</dd></div>
          <div><dt><code>transform.position</code></dt><dd>Active anchor position in normalized units.</dd></div>
          <div><dt><code>transform.pixel_position</code></dt><dd>Active anchor in source-image pixels.</dd></div>
          <div><dt><code>transform.size</code></dt><dd>Normalized width and height.</dd></div>
          <div><dt><code>transform.pixel_size</code></dt><dd>Width and height in pixels.</dd></div>
          <div><dt><code>corner[index]</code></dt><dd>Normalized corner, indexed 1 through 4.</dd></div>
          <div><dt><code>side[index]</code></dt><dd>Normalized side midpoint, indexed 1 through 4.</dd></div>
        </dl>
      </section>
      <section className="doc-section prose">
        <h2 id="methods">Methods</h2>
        <dl className="api-list">{methods.map(([name, detail]) => <div key={name}><dt><code>{name}</code></dt><dd>{detail}</dd></div>)}</dl>
      </section>
      <section className="doc-section prose">
        <h2 id="anchors">Anchor points</h2>
        <div className="anchor-grid" aria-label="Available anchor points">
          {['TOP_LEFT','TOP','TOP_RIGHT','LEFT','CENTER','RIGHT','BOTTOM_LEFT','BOTTOM','BOTTOM_RIGHT'].map((anchor) => <code key={anchor}>{anchor}</code>)}
        </div>
      </section>
      <NextLinks items={[
        { href: "/docs/inference", label: "Back to inference", detail: "Results creation and retained pixels" },
        { href: "/docs/models", label: "Model contract", detail: "Raw outputs and feature scales" },
      ]} />
    </>
  );
}
