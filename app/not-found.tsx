import Link from "next/link";

export default function NotFound() {
  return (
    <main id="content" className="not-found">
      <p className="eyebrow">HTTP 404 / missed detection</p>
      <h1>No box matched this route.</h1>
      <p>The requested page is outside the current documentation graph.</p>
      <Link className="button-primary" href="/docs">Return to documentation →</Link>
    </main>
  );
}
