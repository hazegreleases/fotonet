import { CodeBlock } from "../../ui/CodeBlock";
import { DocHeader, NextLinks, Note } from "../../ui/Docs";
import { ExampleRepositories } from "../../ui/ExampleRepositories";
import { pageMetadata } from "../../seo";

export const metadata = pageMetadata({
  title: "AnchorPoint, Detection Boxes, and Image Transforms",
  description: "Practical fotonet AnchorPoint and BoxTransform documentation with runnable scripts for detection crops, focused subregions, zone gating, containment, and spatial relationships.",
  path: "/docs/transforms",
  keywords: ["AnchorPoint", "bounding box API", "image crop", "region of interest", "coordinate transforms"],
});

const methods = [
  ["set_anchor(anchor)", "Choose the fixed point used by scale and expansion operations. Accepts an AnchorPoint or its string value."],
  ["move((dx, dy))", "Translate the active region in normalized coordinates."],
  ["pixel_move((dx, dy))", "Translate the active region in source-image pixels."],
  ["scale((sx, sy))", "Scale width and height about the active anchor."],
  ["expand(padding)", "Add normalized padding while keeping the active anchor fixed."],
  ["pixel_expand(padding)", "Add source-pixel padding while keeping the active anchor fixed."],
  ["clamp()", "Clamp the active region to normalized image bounds."],
  ["set_box(mode=1)", "Make the region square; mode 1 expands and mode 0 shrinks."],
  ["set_aspect_ratio((w, h), mode=1)", "Fit an aspect ratio around the center; mode 1 expands and mode 0 shrinks."],
  ["crop(image)", "Crop a PIL image or NumPy array using the active region."],
  ["contains(point=...)", "Test a normalized point, or the overlap fraction of this region against another rectangle."],
  ["pixel_contains(point=...)", "Perform the same point or overlap test in source pixels."],
  ["overlaps(other)", "Return whether this transform and another box/transform have non-zero overlap."],
  ["iou(other)", "Compute intersection over union with a DetectionBox or BoxTransform."],
  ["distance(point)", "Measure normalized distance from the active anchor to a point."],
  ["pixel_distance(point)", "Measure source-pixel distance from the active anchor to a point."],
  ["focus(x, y, w, h)", "Replace the active region with a relative subregion inside it."],
  ["focus_reset()", "Restore the active region to the original detection."],
];

const cropCore = `result = model.predict(image_path, conf=0.25, retain_images=True)[0]

for box in result.boxes:
    region = box.transform
    region.set_anchor(AnchorPoint.BOTTOM).pixel_expand(24)
    fixed = region.pixel_position
    region.set_aspect_ratio((4, 5), mode=1)  # preserve the whole region
    shifted = region.pixel_position
    region.pixel_move((fixed.x - shifted.x, fixed.y - shifted.y)).clamp()
    crop = region.crop(result.orig_img)`;

const focusCore = `from fotonet import AnchorPoint, FocusRegion

region = box.transform
upper_body = FocusRegion(x=0.0, y=0.0, w=1.0, h=0.55)
region.focus = upper_body
region.set_anchor(AnchorPoint.CENTER).pixel_expand(12).clamp()
crop = region.crop(result.orig_img)

# The focus is mutable working state; restore the detection when needed.
region.focus_reset()`;

const zoneCore = `image_w, image_h = result.image_size
x1, y1, x2, y2 = zone_pixels
zone = BoxTransform(
    ((x1 + x2) / (2 * image_w),
     (y1 + y2) / (2 * image_h),
     (x2 - x1) / image_w,
     (y2 - y1) / image_h),
    image_size=result.image_size,
).clamp()

for box in result.boxes:
    foot = box.transform.set_anchor(AnchorPoint.BOTTOM).pixel_position
    if zone.pixel_contains(point=foot):
        print(box.cls, box.conf, tuple(foot))`;

export default function TransformsPage() {
  return (
    <>
      <DocHeader eyebrow="Guide / spatial API" title="Turn detections into regions your application can use." lead="AnchorPoint and BoxTransform bridge model output and application geometry: stable crops, subregions, contact-point zones, overlap tests, and coordinate-safe movement in either normalized or pixel units." />

      <section className="doc-section prose">
        <h2 id="mental-model">The original detection and the active region are different things</h2>
        <p><code>DetectionBox.xywh</code> stores the original normalized detection. <code>box.transform</code> starts with the same geometry, then carries mutable working state. Movement, focus, padding, scaling, clamping, and aspect fitting change that active state—not the detector tensor.</p>
        <p>Most mutating transform methods return the transform, so they can be chained. Calling <code>transform.focus(x, y, w, h)</code> is the exception: it returns the applied <code>FocusRegion</code>. Apply focus on one line, then continue the transform chain.</p>
        <Note title="Keep pixels when you need crops"><p>Decoded image paths retain pixels by default. For tensor input, call <code>predict(..., retain_images=True)</code>; otherwise <code>result.orig_img</code> is intentionally unavailable.</p></Note>
      </section>

      <section className="doc-section prose">
        <h2 id="anchors">What an anchor actually fixes</h2>
        <p>An anchor is the point that remains stationary during <code>scale()</code>, <code>expand()</code>, and <code>pixel_expand()</code>. It does not change detection confidence or coordinates by itself.</p>
        <div className="anchor-grid" aria-label="Available anchor points">
          {['TOP_LEFT','TOP','TOP_RIGHT','LEFT','CENTER','RIGHT','BOTTOM_LEFT','BOTTOM','BOTTOM_RIGHT'].map((anchor) => <code key={anchor}>{anchor}</code>)}
        </div>
        <div className="table-wrap comparison-table">
          <table>
            <caption>How common anchors behave during expansion</caption>
            <thead><tr><th>Anchor</th><th>Fixed geometry</th><th>Context grows toward</th><th>Useful for</th></tr></thead>
            <tbody>
              <tr><th><code>CENTER</code></th><td>Box center</td><td>All sides</td><td>Balanced classification crops</td></tr>
              <tr><th><code>TOP</code></th><td>Top midpoint</td><td>Left, right, and down</td><td>Objects hanging from a known top edge</td></tr>
              <tr><th><code>BOTTOM</code></th><td>Bottom midpoint</td><td>Left, right, and up</td><td>People, vehicles, and floor contact</td></tr>
              <tr><th><code>LEFT</code> / <code>RIGHT</code></th><td>Selected side midpoint</td><td>Away from that side</td><td>Edge-aligned UI or scene regions</td></tr>
              <tr><th>Corner anchors</th><td>Selected corner</td><td>Across the opposite quadrant</td><td>Panels and objects attached to image edges</td></tr>
            </tbody>
          </table>
        </div>
        <Note title="Aspect fitting is centered"><p><code>set_box()</code> and <code>set_aspect_ratio()</code> fit around the current center. The downloadable crop scripts preserve a non-center anchor by recording its pixel position, fitting the aspect, and translating the resized region back onto that point. Clamping at an image boundary may still tighten the final ratio.</p></Note>
      </section>

      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Use case 01 / no model required</p><h2 id="manual-region">Compose and inspect a manual region</h2></div>
          <a className="download-button" href="/examples/transform_region.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>This is the smallest complete transform program. Give it an image and normalized <code>cx cy w h</code>; it can fit an aspect ratio, anchor its padding, translate in pixels, clamp, save the crop, and print the final geometry as JSON.</p>
        <CodeBlock code={`python transform_region.py image.jpg \\
  --xywh 0.52 0.48 0.30 0.62 \\
  --aspect 4:5 --anchor bottom --padding 28 \\
  --output outputs/person-card.jpg`} language="bash" label="Run it" />
      </section>

      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Use case 02 / crop dataset</p><h2 id="detection-crops">Extract consistent crops with a manifest</h2></div>
          <a className="download-button" href="/examples/extract_detection_crops.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>The complete script filters exact class names, orders detections by confidence, expands to a requested aspect ratio without cutting the detection, adds anchor-aware pixel context, writes every crop, and records the source and crop geometry in <code>manifest.json</code>.</p>
        <CodeBlock code={cropCore} label="Core transform pipeline" />
        <CodeBlock code={`python extract_detection_crops.py weights/fotonet.pt street.jpg \\
  --class-name person --aspect 4:5 --anchor bottom \\
  --padding 24 --output-dir outputs/people`} language="bash" label="Full-object crops" />
      </section>

      <section className="doc-section prose">
        <h2 id="focus">Focus on a meaningful subregion</h2>
        <p><code>FocusRegion(x, y, w, h)</code> is expressed relative to the active box: <code>(0, 0)</code> is its top-left and <code>(1, 1)</code> reaches its bottom-right. A focus of <code>(0, 0, 1, 0.55)</code> keeps the upper 55% of a person detection—useful for face, torso, badge, or garment pipelines.</p>
        <CodeBlock code={focusCore} label="Upper-body focus" />
        <CodeBlock code={`python extract_detection_crops.py weights/fotonet.pt people.jpg \\
  --class-name person --focus 0 0 1 .55 \\
  --aspect 1:1 --padding 12 --output-dir outputs/upper-body`} language="bash" label="Focused crops" />
        <p><code>FocusRegion.blend(other, alpha)</code> interpolates two subregions. Dividing one focus region by another is shorthand for a 50/50 blend. This is useful for smoothing application presets; it does not track motion over time.</p>
      </section>

      <section className="doc-section prose example-section">
        <div className="example-heading">
          <div><p className="eyebrow">Use case 03 / zone logic</p><h2 id="zone-gating">Gate objects by the anchor that represents contact</h2></div>
          <a className="download-button" href="/examples/anchor_zone_filter.py" download>Download .py <span aria-hidden="true">↓</span></a>
        </div>
        <p>A box center is often the wrong point for floor zones. For a standing person or vehicle, <code>AnchorPoint.BOTTOM</code> approximates the contact point. The script draws the pixel zone, colors accepted/rejected detections, marks each tested anchor, and writes the decision record as JSON.</p>
        <CodeBlock code={zoneCore} label="Bottom-anchor zone test" />
        <CodeBlock code={`python anchor_zone_filter.py weights/fotonet.pt entrance.jpg \\
  --zone 180 240 940 700 --anchor bottom \\
  --class-name person --output outputs/entrance-zone.jpg`} language="bash" label="Run it" />
        <p>For video, <a href="/examples/track_zone_events.py" download>download the tracked zone-event logger</a>. It emits an <code>enter</code> or <code>exit</code> JSON Lines record only when a visible track changes state.</p>
      </section>

      <section className="doc-section prose">
        <h2 id="containment">Point containment and area containment answer different questions</h2>
        <p><code>pixel_contains(point=(x, y))</code> asks whether one point is inside the transform. With <code>x=(x1, x2)</code>, <code>y=(y1, y2)</code>, and <code>threshold</code>, it asks whether the supplied rectangle overlaps at least that fraction of the transform&apos;s own area.</p>
        <CodeBlock code={`# Is the person's bottom anchor inside the zone?
inside_by_contact = zone.pixel_contains(point=foot_point)

# Is at least 75% of the detection covered by the zone rectangle?
mostly_inside = box.transform.pixel_contains(
    x=(zone_x1, zone_x2),
    y=(zone_y1, zone_y2),
    threshold=0.75,
)`} label="Two containment policies" />
        <Note title="Choose the policy deliberately"><p>Anchor containment is stable for entry lines and floor zones. Area containment is stricter and better when most of the visible object must be inside a processing region.</p></Note>
      </section>

      <section className="doc-section prose">
        <h2 id="relationships">Measure relationships between detections</h2>
        <CodeBlock code={`first, second = result.boxes[:2]

overlap = first.transform.overlaps(second)
iou = first.transform.iou(second)

first.transform.set_anchor(AnchorPoint.BOTTOM)
second.transform.set_anchor(AnchorPoint.BOTTOM)
distance_px = first.transform.pixel_distance(second.transform.pixel_position)

print({"overlap": overlap, "iou": iou, "bottom_distance_px": distance_px})`} label="Overlap and anchor distance" />
        <p>IoU is scale-relative and works well for duplicate or nested regions. Pixel distance answers a physical image-space question and depends on resolution. Normalized <code>distance()</code> is more comparable across resized inputs.</p>
      </section>

      <section className="doc-section prose">
        <h2 id="coordinates">Coordinate and point properties</h2>
        <dl className="definition-list">
          <div><dt><code>box.xywh</code></dt><dd>Original normalized detection: center x, center y, width, and height.</dd></div>
          <div><dt><code>box.xyxy</code></dt><dd>Current transform corners. It starts at the detection and changes with mutable transform state.</dd></div>
          <div><dt><code>transform.position</code></dt><dd>Active anchor position in normalized units.</dd></div>
          <div><dt><code>transform.pixel_position</code></dt><dd>Active anchor in source-image pixels.</dd></div>
          <div><dt><code>transform.size</code></dt><dd>Active normalized width and height as <code>Vector2</code>.</dd></div>
          <div><dt><code>transform.pixel_size</code></dt><dd>Active width and height in source pixels.</dd></div>
          <div><dt><code>corner[1..4]</code></dt><dd>Top-left, top-right, bottom-right, and bottom-left normalized corners.</dd></div>
          <div><dt><code>side[1..4]</code></dt><dd>Top, right, bottom, and left normalized side midpoints.</dd></div>
          <div><dt><code>pixel_corner</code> / <code>pixel_side</code></dt><dd>Pixel-space forms using the same clockwise indexing.</dd></div>
        </dl>
        <p><code>Vector2</code> supports <code>.x</code>, <code>.y</code>, tuple unpacking, and indices <code>[0]</code>/<code>[1]</code>. The <code>image_size</code> constructor argument is always <code>(width, height)</code>.</p>
      </section>

      <section className="doc-section prose">
        <h2 id="methods">Complete method reference</h2>
        <dl className="api-list">{methods.map(([name, detail]) => <div key={name}><dt><code>{name}</code></dt><dd>{detail}</dd></div>)}</dl>
      </section>

      <section className="doc-section prose">
        <h2 id="safe-order">A reliable transform order</h2>
        <ol className="numbered-steps">
          <li><b>Start</b><span>Take a fresh <code>DetectionBox.transform</code>, or construct <code>BoxTransform(xywh, image_size)</code>.</span></li>
          <li><b>Focus</b><span>Apply a relative subregion first if the application needs only part of the detection.</span></li>
          <li><b>Anchor</b><span>Select the physical point or edge that should remain stationary, then scale or add directional padding.</span></li>
          <li><b>Shape</b><span>Fit a square or aspect ratio with expand mode; record and restore a non-center anchor when both constraints matter.</span></li>
          <li><b>Move</b><span>Apply any normalized or pixel translation required by the application.</span></li>
          <li><b>Clamp</b><span>Clamp last so saved geometry and extracted pixels agree at image boundaries.</span></li>
        </ol>
      </section>

      <ExampleRepositories topic="transforms" />
      <NextLinks items={[
        { href: "/docs/examples", label: "Download every example", detail: "Complete command-line scripts and run commands" },
        { href: "/docs/inference", label: "Back to inference", detail: "Results creation, filtering, and retained pixels" },
      ]} />
    </>
  );
}
