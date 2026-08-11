import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const output = join(root, "pages-dist");
const routes = [
  "", "benchmarks", "docs", "docs/api", "docs/checkpoints", "docs/cli",
  "docs/data", "docs/examples", "docs/export", "docs/inference", "docs/install",
  "docs/models", "docs/training", "docs/transforms", "docs/validation",
];

for (const route of routes) {
  const html = await readFile(join(output, route, "index.html"), "utf8");
  assert.match(html, /fotonet/i, route || "/");
  assert.doesNotMatch(html, /chatgpt\.site|dev-tools\//i, route || "/");
  assert.doesNotMatch(html, /from fotonet import FOTONET/, route || "/");
  assert.doesNotMatch(html, /<script(?![^>]*static-runtime\.js)/i, route || "/");
  assert.doesNotMatch(html, /(?:href|src)="\/(?!fotonet\/)/i, route || "/");
  assert.match(html, /\/fotonet\/static-runtime\.js/, route || "/");
}

const examples = await readFile(join(output, "docs", "examples", "index.html"), "utf8");
assert.match(examples, /\/fotonet\/examples\/train\.py/);
await access(join(output, "examples", "train.py"));
await access(join(output, ".nojekyll"));
await access(join(output, "404.html"));
console.log(`Verified ${routes.length} static routes and all downloadable assets.`);
