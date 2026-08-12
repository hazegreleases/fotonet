import type { Metadata, Viewport } from "next";
import "./globals.css";
import { projectStructuredData, SITE_URL } from "./seo";
import { SiteFooter, SiteHeader } from "./ui/SiteChrome";

const title = "fotonet: Compact NMS-Free Object Detection in PyTorch";
const description = "Documentation for fotonet, a compact NMS-free PyTorch object detector with resumable training, COCO validation, Python APIs, and ONNX or TorchScript export.";

export const metadata: Metadata = {
  metadataBase: new URL(`${SITE_URL}/`),
  title: { default: title, template: "%s | fotonet" },
  description,
  applicationName: "fotonet",
  category: "computer vision",
  keywords: [
    "fotonet", "FOTO-NET", "object detection", "computer vision",
    "machine learning", "deep learning", "artificial intelligence",
    "PyTorch", "NMS-free detection", "real-time object detection",
  ],
  authors: [{ name: "FOTO-NET contributors", url: "https://github.com/hazegreleases/fotonet" }],
  creator: "FOTO-NET contributors",
  publisher: "FOTO-NET contributors",
  alternates: {
    canonical: `${SITE_URL}/`,
    types: { "application/xml": `${SITE_URL}/sitemap.xml` },
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
      "max-video-preview": -1,
    },
  },
  manifest: "/manifest.webmanifest",
  icons: { icon: "/favicon.svg" },
  openGraph: {
    title,
    description,
    url: `${SITE_URL}/`,
    siteName: "fotonet",
    locale: "en_US",
    type: "website",
    images: [{ url: `${SITE_URL}/og.png`, width: 1200, height: 630, alt: "fotonet — compact object detection, measured and documented" }],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: [`${SITE_URL}/og.png`],
  },
};

export const viewport: Viewport = {
  colorScheme: "light dark",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#e9e0cd" },
    { media: "(prefers-color-scheme: dark)", color: "#09110c" },
  ],
};

const systemColorModeScript = `(() => { try { const dark = window.matchMedia('(prefers-color-scheme: dark)').matches; document.documentElement.dataset.colorMode = dark ? 'dark' : 'light'; } catch (_) { document.documentElement.dataset.colorMode = 'light'; } })();`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script id="fotonet-color-mode" dangerouslySetInnerHTML={{ __html: systemColorModeScript }} />
        <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(projectStructuredData) }} />
      </head>
      <body>
        <SiteHeader />
        {children}
        <SiteFooter />
      </body>
    </html>
  );
}
