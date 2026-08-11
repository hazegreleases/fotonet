"use client";

import { useState } from "react";

export function CodeBlock({ code, language = "python", label }: { code: string; language?: string; label?: string }) {
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");

  function fallbackCopy() {
    const textarea = document.createElement("textarea");
    textarea.value = code;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    if (!copied) throw new Error("Copy command was rejected");
  }

  async function copy() {
    try {
      if (window.isSecureContext && navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(code);
        } catch {
          fallbackCopy();
        }
      } else {
        fallbackCopy();
      }
      setCopyState("copied");
    } catch {
      setCopyState("failed");
    }
    window.setTimeout(() => setCopyState("idle"), 1800);
  }

  const buttonLabel = copyState === "copied" ? "Copied" : copyState === "failed" ? "Select code" : "Copy";
  const lines = code.split("\n");

  return (
    <figure className="code-block">
      <figcaption>
        <span className="code-window-dots" aria-hidden="true"><i /><i /><i /></span>
        <span className="code-label">{label ?? language}</span>
        <span className="code-language">{language}</span>
        <button type="button" onClick={copy} aria-live="polite">{buttonLabel}</button>
      </figcaption>
      <pre data-language={language}>
        <code>{lines.map((line, index) => (
          <span className="code-line" key={`${index}-${line}`}>
            <span className="line-number" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
            <span className="line-source">{line || " "}</span>
          </span>
        ))}</code>
      </pre>
    </figure>
  );
}
