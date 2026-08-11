"use client";

import { useEffect, useState } from "react";

const themes = [
  { id: "field", label: "Field" },
  { id: "night", label: "Night lab" },
  { id: "blueprint", label: "Blueprint" },
] as const;

type Theme = (typeof themes)[number]["id"];

function applyTheme(next: Theme) {
  document.documentElement.dataset.theme = next;
  window.localStorage.setItem("fotonet-theme", next);
}

export function ThemeSwitch() {
  const [theme, setTheme] = useState<Theme>("field");

  useEffect(() => {
    const stored = window.localStorage.getItem("fotonet-theme") as Theme | null;
    const next = themes.some((item) => item.id === stored) ? stored! : "field";
    applyTheme(next);
    const timer = window.setTimeout(() => setTheme(next), 0);
    return () => window.clearTimeout(timer);
  }, []);

  function choose(next: Theme) {
    applyTheme(next);
    setTheme(next);
  }

  return (
    <div className="theme-switch" aria-label="Color theme">
      {themes.map((item) => (
        <button
          type="button"
          key={item.id}
          aria-pressed={theme === item.id}
          aria-label={`Use ${item.label} color theme`}
          onClick={() => choose(item.id)}
        >
          <span className={`theme-dot theme-dot-${item.id}`} aria-hidden="true" />
          <span className="theme-label">{item.label}</span>
        </button>
      ))}
    </div>
  );
}
