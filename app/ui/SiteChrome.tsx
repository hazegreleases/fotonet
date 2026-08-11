import Link from "next/link";
import { primaryNav } from "../data";
import { ThemeSwitch } from "./ThemeSwitch";

export function SiteHeader() {
  return (
    <header className="site-header">
      <a className="skip-link" href="#content">Skip to content</a>
      <div className="header-inner">
        <Link className="wordmark" href="/" aria-label="FOTO-NET home">
          <span className="mark" aria-hidden="true">F:</span>
          <span>fotonet</span>
          <small>0.8.0b2</small>
        </Link>
        <nav className="primary-nav" aria-label="Primary navigation">
          {primaryNav.map((item) => item.external ? (
            <a key={item.href} href={item.href} target="_blank" rel="noreferrer">{item.label}<span aria-hidden="true">↗</span></a>
          ) : (
            <Link key={item.href} href={item.href}>{item.label}</Link>
          ))}
        </nav>
        <ThemeSwitch />
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div>
        <span className="footer-mark">F:</span>
        <p>Lightweight, NMS-free object detection.<br />The first official Nano weight is training now.</p>
      </div>
      <div className="footer-links">
        <Link href="/docs">Documentation</Link>
        <Link href="/benchmarks">Benchmarks</Link>
        <a href="https://github.com/hazegreleases/fotonet" target="_blank" rel="noreferrer">Source code ↗</a>
        <a href="https://github.com/hazegreleases/fotonet/blob/main/LICENSE" target="_blank" rel="noreferrer">Apache-2.0 ↗</a>
      </div>
      <p className="footer-note">OFFICIAL WEIGHT IN TRAINING · RELEASE + SHA256 TO FOLLOW · NO AP CLAIM YET · PYTHON 3.10+</p>
    </footer>
  );
}
