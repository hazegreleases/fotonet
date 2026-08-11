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
const titles = new Set();
const descriptions = new Set();

for (const route of routes) {
  const html = await readFile(join(output, route, "index.html"), "utf8");
  assert.match(html, /fotonet/i, route || "/");
  assert.doesNotMatch(html, /chatgpt\.site|dev-tools\//i, route || "/");
  assert.doesNotMatch(html, /from fotonet import FOTONET/, route || "/");
  assert.doesNotMatch(html, /<script(?![^>]*(?:static-runtime\.js|application\/ld\+json))/i, route || "/");
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
await access(join(output, "examples", "train.py"));
await access(join(output, ".nojekyll"));
await access(join(output, "404.html"));
const home = await readFile(join(output, "index.html"), "utf8");
assert.match(home, /"@type":"WebSite"/);
assert.match(home, /"@type":"SoftwareSourceCode"/);
assert.match(home, /href="\/fotonet\/favicon\.svg"/);
const sitemap = await readFile(join(output, "sitemap.xml"), "utf8");
assert.equal((sitemap.match(/<url>/g) ?? []).length, routes.length);
assert.match(sitemap, /https:\/\/hazegreleases\.github\.io\/fotonet\/docs\/training\//);
const robots = await readFile(join(output, "robots.txt"), "utf8");
assert.match(robots, /Sitemap: https:\/\/hazegreleases\.github\.io\/fotonet\/sitemap\.xml/);
console.log(`Verified ${routes.length} static routes and all downloadable assets.`);
