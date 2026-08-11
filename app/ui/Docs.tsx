import Link from "next/link";
import { docsNav } from "../data";
import { AmbientShapes } from "./AmbientShapes";

export function DocsSidebar() {
  return (
    <aside className="docs-sidebar" aria-label="Documentation navigation">
      <p className="sidebar-title">Manual / v0.8.0b2</p>
      {docsNav.map((group) => (
        <section key={group.label}>
          <h2>{group.label}</h2>
          <ul>
            {group.items.map((item) => <li key={item.href}><Link href={item.href}>{item.label}</Link></li>)}
          </ul>
        </section>
      ))}
    </aside>
  );
}

export function DocHeader({ eyebrow, title, lead }: { eyebrow: string; title: string; lead: string }) {
  return (
    <header className="doc-header">
      <AmbientShapes variant="document" />
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <p className="doc-lead">{lead}</p>
    </header>
  );
}

export function Note({ title, children, tone = "note" }: { title: string; children: React.ReactNode; tone?: "note" | "warning" | "success" }) {
  return (
    <aside className={`note note-${tone}`}>
      <strong>{title}</strong>
      <div>{children}</div>
    </aside>
  );
}

export function NextLinks({ items }: { items: { href: string; label: string; detail: string }[] }) {
  return (
    <nav className="next-links" aria-label="Related documentation">
      {items.map((item) => (
        <Link key={item.href} href={item.href}>
          <span>{item.label}</span>
          <small>{item.detail}</small>
        </Link>
      ))}
    </nav>
  );
}
