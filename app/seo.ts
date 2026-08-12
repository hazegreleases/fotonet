import type { Metadata } from "next";

export const SITE_URL = "https://hazegreleases.github.io/fotonet";

const BASE_KEYWORDS = [
  "fotonet",
  "FOTO-NET",
  "object detection",
  "computer vision",
  "machine learning",
  "deep learning",
  "artificial intelligence",
  "PyTorch",
  "NMS-free detector",
];

type PageMetadata = {
  title: string;
  description: string;
  path: string;
  keywords?: string[];
};

function absoluteUrl(path: string) {
  return path === "/" ? `${SITE_URL}/` : `${SITE_URL}${path}/`;
}

export function pageMetadata({ title, description, path, keywords = [] }: PageMetadata): Metadata {
  const url = absoluteUrl(path);
  const image = `${SITE_URL}/og.png`;
  return {
    title,
    description,
    keywords: [...BASE_KEYWORDS, ...keywords],
    alternates: { canonical: url },
    openGraph: {
      title: `${title} | fotonet`,
      description,
      url,
      siteName: "fotonet",
      locale: "en_US",
      type: "website",
      images: [{ url: image, width: 1200, height: 630, alt: "fotonet — compact object detection, measured and documented" }],
    },
    twitter: {
      card: "summary_large_image",
      title: `${title} | fotonet`,
      description,
      images: [image],
    },
  };
}

export const projectStructuredData = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebSite",
      "@id": `${SITE_URL}/#website`,
      url: `${SITE_URL}/`,
      name: "fotonet",
      alternateName: "FOTO-NET",
      description: "Documentation for a compact NMS-free PyTorch object detector.",
      inLanguage: "en",
    },
    {
      "@type": "SoftwareSourceCode",
      "@id": `${SITE_URL}/#software`,
      name: "fotonet",
      alternateName: "FOTO-NET",
      version: "0.8.0b2",
      description: "A compact NMS-free object detection library for PyTorch with training, COCO validation, Python inference APIs, and deployment export.",
      url: `${SITE_URL}/`,
      codeRepository: "https://github.com/hazegreleases/fotonet",
      downloadUrl: "https://pypi.org/project/fotonet/",
      license: "https://www.apache.org/licenses/LICENSE-2.0",
      programmingLanguage: {
        "@type": "ComputerLanguage",
        name: "Python",
      },
      runtimePlatform: "PyTorch",
      keywords: "object detection, computer vision, machine learning, deep learning, NMS-free detection, PyTorch, ONNX",
    },
  ],
};
