import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";
import { SiteFooter, SiteHeader } from "./ui/SiteChrome";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const title = "fotonet — production V1 documentation";
  const description = "Complete documentation for the fotonet production V1 detector: datasets, training, resume, checkpoints, inference, validation, exports, API contracts, and measured deployment performance.";
  return {
    title: { default: "fotonet — compact object detection", template: "%s · fotonet" },
    description,
    metadataBase: new URL(origin),
    openGraph: {
      title,
      description,
      type: "website",
      images: [{ url: `${origin}/og.png`, width: 1739, height: 909, alt: "fotonet — Small detector. Explicit contracts." }],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [`${origin}/og.png`],
    },
  };
}

const themeScript = `(() => { try { const value = localStorage.getItem('fotonet-theme'); document.documentElement.dataset.theme = ['field','night','blueprint'].includes(value) ? value : 'field'; } catch (_) { document.documentElement.dataset.theme = 'field'; } })();`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-theme="field" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: themeScript }} /></head>
      <body>
        <SiteHeader />
        {children}
        <SiteFooter />
      </body>
    </html>
  );
}
