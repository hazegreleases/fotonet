import { DocsSidebar } from "../ui/Docs";

export default function DocumentationLayout({ children }: { children: React.ReactNode }) {
  return (
    <main id="content" className="docs-shell">
      <DocsSidebar />
      <article className="doc-content">{children}</article>
    </main>
  );
}
