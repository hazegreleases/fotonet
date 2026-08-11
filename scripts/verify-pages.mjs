import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const output = join(root, "pages-dist");
const repositoryTopics = [
  "inference", "data", "training", "validation", "export",
  "examples", "models", "checkpoints", "api", "transforms",
];
const projectRoutes = repositoryTopics.flatMap((topic) =>
  [1, 2, 3].map((tier) => `code/${topic}/tier-${tier}`),
);
const routes = [
  "", "benchmarks", "docs", "docs/api", "docs/checkpoints", "docs/cli",
  "docs/data", "docs/examples", "docs/export", "docs/inference", "docs/install",
  "docs/models", "docs/training", "docs/transforms", "docs/validation",
  ...projectRoutes,
];
const titles = new Set();
const descriptions = new Set();

for (const route of routes) {
  const html = await readFile(join(output, route, "index.html"), "utf8");
  assert.match(html, /fotonet/i, route || "/");
  assert.doesNotMatch(html, /chatgpt\.site|dev-tools\//i, route || "/");
  assert.doesNotMatch(html, /from fotonet import FOTONET/, route || "/");
  assert.doesNotMatch(html, /theme-switch|data-theme|fotonet-theme|blueprint/i, route || "/");
  assert.doesNotMatch(html, /<script(?![^>]*(?:static-runtime\.js|application\/ld\+json|fotonet-color-mode|data-project-files))/i, route || "/");
  assert.match(html, /<script id="fotonet-color-mode">/, route || "/");
  assert.doesNotMatch(html, /(?:href|src)="\/(?!fotonet\/)/i, route || "/");
  assert.match(html, /\/fotonet\/static-runtime\.js/, route || "/");
  assert.match(html, /<meta name="description" content="[^"]{80,}/i, route || "/");
  assert.match(html, /<link rel="canonical" href="https:\/\/hazegreleases\.github\.io\/fotonet\//i, route || "/");
  assert.match(html, /<meta property="og:title"/i, route || "/");
  const title = html.match(/<title>([^<]+)<\/title>/i)?.[1];
  const description = html.match(/<meta name="description" content="([^"]+)"/i)?.[1];
  assert.ok(title, `missing title: ${route || "/"}`);
  assert.ok(description, `missing description: ${route || "/"}`);
  assert.ok(!titles.has(title), `duplicate title: ${title}`);
  assert.ok(!descriptions.has(description), `duplicate description: ${description}`);
  titles.add(title);
  descriptions.add(description);
}

const examples = await readFile(join(output, "docs", "examples", "index.html"), "utf8");
assert.match(examples, /\/fotonet\/examples\/train\.py/);
for (const script of [
  "train.py", "predict_image.py", "predict_folder.py", "export_onnx.py",
  "validate_checkpoint.py", "transform_region.py", "extract_detection_crops.py",
  "anchor_zone_filter.py", "track_zone_events.py",
]) {
  assert.match(examples, new RegExp(`/fotonet/examples/${script.replace(".", "\\.")}`));
  await access(join(output, "examples", script));
}
const transforms = await readFile(join(output, "docs", "transforms", "index.html"), "utf8");
assert.match(transforms, /AnchorPoint\.BOTTOM/);
assert.match(transforms, /pixel_contains/);
assert.match(transforms, /focus_reset/);
const transformProject = await readFile(join(output, "code", "transforms", "tier-3", "index.html"), "utf8");
assert.match(transformProject, />11<\/dd>/);
assert.match(transformProject, /<dt>Source<\/dt>/);
assert.match(transformProject, /<dt>Average<\/dt>/);
assert.match(transformProject, /\/fotonet\/examples\/projects\/transform-zone-system\.zip/);
const allRounderProject = await readFile(join(output, "code", "examples", "tier-3", "index.html"), "utf8");
assert.match(allRounderProject, />30<\/dd>/);
assert.match(allRounderProject, /<dt>Source<\/dt>/);
assert.match(allRounderProject, /<dt>Average<\/dt>/);
assert.match(allRounderProject, /\/fotonet\/examples\/projects\/fotonet-all-rounder\.zip/);
for (const [project, expectedFiles, minimumAverage, maximumAverage] of [
  ["transform-zone-system", 11, 65, 80],
  ["fotonet-all-rounder", 30, 140, 160],
]) {
  const projectRoot = join(output, "examples", "projects", project);
  const manifest = JSON.parse(await readFile(join(projectRoot, "manifest.json"), "utf8"));
  assert.equal(manifest.files.length, expectedFiles);
  const averageLines = manifest.files.reduce((total, file) => total + file.lines, 0) / manifest.files.length;
  assert.ok(averageLines >= minimumAverage && averageLines <= maximumAverage, `${project}: ${averageLines}`);
  for (const file of manifest.files) await access(join(projectRoot, file.path));
  await access(join(output, "examples", "projects", `${project}.zip`));
}
for (const topic of repositoryTopics) {
  const guide = await readFile(join(output, "docs", topic, "index.html"), "utf8");
  for (const tier of [1, 2, 3]) {
    assert.match(guide, new RegExp(`/fotonet/code/${topic}/tier-${tier}`));
    const project = await readFile(join(output, "code", topic, `tier-${tier}`, "index.html"), "utf8");
    assert.match(project, /data-project-workspace/);
    assert.match(project, /data-project-file="0"/);
    assert.match(project, /class="project-code-viewport"/);
  }
}
await access(join(output, ".nojekyll"));
await access(join(output, "404.html"));
const home = await readFile(join(output, "index.html"), "utf8");
assert.match(home, /"@type":"WebSite"/);
assert.match(home, /"@type":"SoftwareSourceCode"/);
assert.match(home, /href="\/fotonet\/favicon\.svg"/);
assert.match(home, /<meta name="viewport" content="width=device-width, initial-scale=1"/);
assert.match(home, /class="mobile-nav"/);
assert.equal((home.match(/name="theme-color"/g) ?? []).length, 2);
const colorModeScript = home.match(/<script id="fotonet-color-mode">([\s\S]*?)<\/script>/)?.[1];
assert.ok(colorModeScript, "missing pre-paint system color-mode script");
assert.ok(home.indexOf("fotonet-color-mode") < home.indexOf("</head>"), "color-mode script must run before body paint");
for (const dark of [false, true]) {
  const document = { documentElement: { dataset: {} } };
  const window = { matchMedia: () => ({ matches: dark }) };
  Function("window", "document", colorModeScript)(window, document);
  assert.equal(document.documentElement.dataset.colorMode, dark ? "dark" : "light");
}
const cssHref = home.match(/href="\/fotonet\/(?<path>_next\/static\/css\/[^"]+\.css)"/)?.groups?.path;
assert.ok(cssHref, "missing compiled stylesheet");
const css = await readFile(join(output, cssHref), "utf8");
assert.match(css, /data-color-mode=dark/);
assert.match(css, /prefers-color-scheme:\s*dark/);
assert.doesNotMatch(css, /data-theme|theme-switch|blueprint/i);
const documentation = await readFile(join(output, "docs", "index.html"), "utf8");
assert.match(documentation, /class="docs-nav-disclosure"/);
const sitemap = await readFile(join(output, "sitemap.xml"), "utf8");
assert.equal((sitemap.match(/<url>/g) ?? []).length, routes.length);
assert.match(sitemap, /https:\/\/hazegreleases\.github\.io\/fotonet\/docs\/training\//);
const robots = await readFile(join(output, "robots.txt"), "utf8");
assert.match(robots, /Sitemap: https:\/\/hazegreleases\.github\.io\/fotonet\/sitemap\.xml/);
console.log(`Verified ${routes.length} static routes and all downloadable assets.`);
