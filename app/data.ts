export const primaryNav = [
  { href: "/docs", label: "Documentation" },
  { href: "/docs/models", label: "Models" },
  { href: "/benchmarks", label: "Benchmarks" },
  { href: "https://github.com/hazegreleases/fotonet", label: "GitHub", external: true },
] as const;

export const docsNav = [
  {
    label: "Start",
    items: [
      { href: "/docs", label: "Documentation map" },
      { href: "/docs/install", label: "Install & first run" },
      { href: "/docs/cli", label: "Command line" },
    ],
  },
  {
    label: "Workflows",
    items: [
      { href: "/docs/inference", label: "Inference" },
      { href: "/docs/data", label: "Datasets & labels" },
      { href: "/docs/training", label: "Training & resume" },
      { href: "/docs/validation", label: "Validation" },
      { href: "/docs/export", label: "Export" },
      { href: "/docs/examples", label: "Runnable examples" },
    ],
  },
  {
    label: "Reference",
    items: [
      { href: "/docs/models", label: "Models & configuration" },
      { href: "/docs/checkpoints", label: "Checkpoint contracts" },
      { href: "/docs/api", label: "Python API" },
      { href: "/docs/transforms", label: "Results & transforms" },
      { href: "/benchmarks", label: "Measured performance" },
    ],
  },
] as const;

export type ModelMetric = {
  name: string;
  p2: boolean;
  fullParams: string;
  deployParams: string;
  gmac: string;
  gflops: string;
  fps1: string;
  fps8: string;
  latency1: string;
  vram1: string;
  vram8: string;
  candidates: string;
};

export const modelMetrics: ModelMetric[] = [
  { name: "fotonetn", p2: false, fullParams: "1.043M", deployParams: "1.006M", gmac: "1.313", gflops: "2.626", fps1: "218.33", fps8: "723.43", latency1: "4.225", vram1: "30", vram8: "218", candidates: "8,400" },
  { name: "fotonetn-p2", p2: true, fullParams: "1.082M", deployParams: "1.036M", gmac: "1.691", gflops: "3.381", fps1: "154.31", fps8: "281.91", latency1: "6.338", vram1: "94", vram8: "730", candidates: "34,000" },
  { name: "fotonets", p2: false, fullParams: "1.516M", deployParams: "1.479M", gmac: "2.138", gflops: "4.275", fps1: "191.57", fps8: "460.46", latency1: "4.664", vram1: "41", vram8: "276", candidates: "8,400" },
  { name: "fotonets-p2", p2: true, fullParams: "1.568M", deployParams: "1.526M", gmac: "2.664", gflops: "5.328", fps1: "152.78", fps8: "240.25", latency1: "6.306", vram1: "99", vram8: "755", candidates: "34,000" },
  { name: "fotonetm", p2: false, fullParams: "2.973M", deployParams: "2.934M", gmac: "3.696", gflops: "7.392", fps1: "174.89", fps8: "346.32", latency1: "5.431", vram1: "51", vram8: "308", candidates: "8,400" },
  { name: "fotonetm-p2", p2: true, fullParams: "3.064M", deployParams: "3.020M", gmac: "4.615", gflops: "9.230", fps1: "139.88", fps8: "189.77", latency1: "6.846", vram1: "110", vram8: "800", candidates: "34,000" },
  { name: "fotonetl", p2: false, fullParams: "5.175M", deployParams: "5.132M", gmac: "6.606", gflops: "13.213", fps1: "169.75", fps8: "228.88", latency1: "5.388", vram1: "68", vram8: "394", candidates: "8,400" },
  { name: "fotonetl-p2", p2: true, fullParams: "5.327M", deployParams: "5.279M", gmac: "8.116", gflops: "16.232", fps1: "137.33", fps8: "139.11", latency1: "7.051", vram1: "125", vram8: "852", candidates: "34,000" },
  { name: "fotonetx", p2: false, fullParams: "8.690M", deployParams: "8.644M", gmac: "10.819", gflops: "21.638", fps1: "144.01", fps8: "157.77", latency1: "6.581", vram1: "92", vram8: "485", candidates: "8,400" },
  { name: "fotonetx-p2", p2: true, fullParams: "8.921M", deployParams: "8.869M", gmac: "13.079", gflops: "26.159", fps1: "106.40", fps8: "102.17", latency1: "9.046", vram1: "145", vram8: "914", candidates: "34,000" },
];

export const installCode = `python -m pip install fotonet`;

export const inferenceCode = `from fotonet import Fotonet

model = Fotonet("path/to/checkpoint.pt")
results = model.predict("image.jpg", conf=0.25, imgsz=640)

for box in results[0].boxes:
    print(box.cls, box.conf, box.xyxy)`;

export const trainCode = `from fotonet import Fotonet

model = Fotonet("fotonetn", nc=2)
summary = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)`;

export const launcherCode = `python train.py \\
  --model fotonetn \\
  --data path/to/data.yaml \\
  --epochs 300 \\
  --batch 16 \\
  --run-dir runs/fotonetn`;

export const resumeCode = `python train.py \\
  --model fotonetn \\
  --data path/to/data.yaml \\
  --epochs 300 \\
  --batch 16 \\
  --run-dir runs/fotonetn \\
  --resume`;

export const modelGuidance = [
  { profile: "N", role: "Lowest deployment cost", standard: "fotonetn", p2: "fotonetn-p2" },
  { profile: "S", role: "Small model with more channel capacity", standard: "fotonets", p2: "fotonets-p2" },
  { profile: "M", role: "Middle of the current family", standard: "fotonetm", p2: "fotonetm-p2" },
  { profile: "L", role: "Higher-capacity deployment", standard: "fotonetl", p2: "fotonetl-p2" },
  { profile: "X", role: "Largest stable graph", standard: "fotonetx", p2: "fotonetx-p2" },
] as const;
