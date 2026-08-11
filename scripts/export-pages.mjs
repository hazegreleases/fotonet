import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const output = join(root, "pages-dist");
const basePath = (process.env.PAGES_BASE_PATH ?? "/fotonet").replace(/\/$/, "");
const origin = "https://hazegreleases.github.io";
const routes = [
  "/", "/benchmarks", "/docs", "/docs/api", "/docs/checkpoints",
  "/docs/cli", "/docs/data", "/docs/examples", "/docs/export",
  "/docs/inference", "/docs/install", "/docs/models", "/docs/training",
  "/docs/transforms", "/docs/validation",
];

await rm(output, { recursive: true, force: true });
await cp(join(root, "dist", "client"), output, { recursive: true });

const workerUrl = pathToFileURL(join(root, "dist", "server", "index.js"));
workerUrl.searchParams.set("static-export", String(Date.now()));
const { default: worker } = await import(workerUrl.href);
const render = typeof worker === "function"
  ? (request) => worker(request)
  : (request) => worker.fetch(
      request,
      { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
      { waitUntil() {}, passThroughOnException() {} },
    );

function makeStatic(html) {
  const withoutRuntime = html
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<link\b[^>]*rel="modulepreload"[^>]*>/gi, "")
    .replace(/\sdata-rsc-css-href="[^"]*"/gi, "");
  const based = withoutRuntime.replace(
    /\b(href|src)="\/(?!\/)/g,
    (_match, attribute) => `${attribute}="${basePath}/`,
  );
  return based.replace(
    "</body>",
    `<script defer src="${basePath}/static-runtime.js"></script></body>`,
  );
}

for (const route of routes) {
  const response = await render(
    new Request(`${origin}${route}`, { headers: { accept: "text/html" } }),
  );
  if (!response.ok) throw new Error(`Render failed for ${route}: HTTP ${response.status}`);
  const destination = route === "/"
    ? join(output, "index.html")
    : join(output, route.slice(1), "index.html");
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(destination, makeStatic(await response.text()), "utf8");
}

await writeFile(join(output, ".nojekyll"), "", "utf8");
const home = await readFile(join(output, "index.html"), "utf8");
await writeFile(join(output, "404.html"), home, "utf8");
console.log(`Exported ${routes.length} routes to ${output} with base path ${basePath}`);
