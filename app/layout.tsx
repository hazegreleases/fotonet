import type { Metadata } from "next";
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
    images: [{ url: `${SITE_URL}/og.png`, width: 1739, height: 909, alt: "fotonet compact object detection documentation" }],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: [`${SITE_URL}/og.png`],
  },
};

const themeScript = `(() => { try { const value = localStorage.getItem('fotonet-theme'); document.documentElement.dataset.theme = ['field','night','blueprint'].includes(value) ? value : 'field'; } catch (_) { document.documentElement.dataset.theme = 'field'; } })();`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-theme="field" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
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
