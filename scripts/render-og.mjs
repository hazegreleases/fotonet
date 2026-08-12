import { existsSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "public", "og-source.svg");
const output = join(root, "public", "og.png");

const candidates = [
  process.env.FOTONET_CHROME,
  process.platform === "win32" && "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  process.platform === "win32" && "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  process.platform === "darwin" && "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  process.platform === "linux" && "/usr/bin/google-chrome",
  process.platform === "linux" && "/usr/bin/chromium",
].filter(Boolean);

const chrome = candidates.find((candidate) => existsSync(candidate));
if (!chrome) {
  throw new Error("Chrome was not found. Set FOTONET_CHROME to its executable path.");
}

const result = spawnSync(chrome, [
  "--headless=new",
  "--disable-gpu",
  "--disable-software-rasterizer=false",
  "--force-device-scale-factor=1",
  "--hide-scrollbars",
  "--window-size=1200,630",
  `--screenshot=${output}`,
  pathToFileURL(source).href,
], { encoding: "utf8" });

if (result.status !== 0) {
  throw new Error(result.stderr || result.stdout || `Chrome exited with ${result.status}`);
}

const png = readFileSync(output);
const signature = png.subarray(0, 8).toString("hex");
const width = png.readUInt32BE(16);
const height = png.readUInt32BE(20);
if (signature !== "89504e470d0a1a0a" || width !== 1200 || height !== 630) {
  throw new Error(`Expected a 1200x630 PNG, received ${width}x${height}.`);
}

console.log(`Rendered ${output} from ${source} (${width}x${height}).`);
