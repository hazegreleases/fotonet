"use client";

import { useEffect, useRef } from "react";

type ShapeVariant = "hero" | "document" | "technical" | "organic";

export function AmbientShapes({ variant }: { variant: ShapeVariant }) {
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = root.current;
    if (!node) return;

    if (!("IntersectionObserver" in window)) {
      node.classList.add("is-seen");
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        node.classList.add("is-seen");
        observer.unobserve(node);
      },
      { threshold: 0.12 },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={root} className={`ambient-shapes ambient-${variant}`} aria-hidden="true">
      <span className="ambient-shape shape-one" />
      <span className="ambient-shape shape-two" />
      <span className="ambient-shape shape-three" />
      <span className="ambient-shape shape-four" />
      <span className="ambient-shape shape-five" />
      <span className="ambient-shape shape-six" />
      <span className="ambient-shape shape-seven" />
    </div>
  );
}
