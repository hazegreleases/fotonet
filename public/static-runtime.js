(() => {
  const systemColorMode = window.matchMedia("(prefers-color-scheme: dark)");
  const syncColorMode = () => {
    document.documentElement.dataset.colorMode = systemColorMode.matches ? "dark" : "light";
  };
  syncColorMode();
  if (typeof systemColorMode.addEventListener === "function") {
    systemColorMode.addEventListener("change", syncColorMode);
  } else if (typeof systemColorMode.addListener === "function") {
    systemColorMode.addListener(syncColorMode);
  }

  document.addEventListener("click", async (event) => {
    if (!(event.target instanceof Element)) return;
    const button = event.target.closest(".code-block figcaption button");
    if (!button) return;
    const block = button.closest(".code-block");
    const lines = [...block.querySelectorAll(".line-source")];
    if (lines.length === 0) return;
    const source = lines.map((line) => line.textContent ?? "").join("\n");
    const fallbackCopy = () => {
      const textarea = document.createElement("textarea");
      textarea.value = source;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      textarea.remove();
      if (!copied) throw new Error("Copy command was rejected");
    };
    try {
      if (window.isSecureContext && navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(source);
        } catch {
          fallbackCopy();
        }
      } else {
        fallbackCopy();
      }
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select code";
    }
    window.setTimeout(() => { button.textContent = "Copy"; }, 1800);
  });

  document.querySelectorAll("[data-project-workspace]").forEach((workspace) => {
    const files = [...workspace.querySelectorAll("[data-project-file]")];
    const panel = workspace.querySelector("[data-project-panel]");
    const data = workspace.querySelector("[data-project-files]");
    if (files.length === 0 || !panel || !data) return;
    let projectFiles;
    try {
      projectFiles = JSON.parse(data.textContent ?? "[]");
    } catch {
      return;
    }
    if (!Array.isArray(projectFiles) || projectFiles.length !== files.length) return;

    const renderFile = (file, index) => {
      panel.dataset.projectPanel = String(index);
      const path = panel.querySelector(".project-editor-path");
      const pathLabel = path.querySelector("strong");
      pathLabel.textContent = file.name;
      let download = path.querySelector("a");
      if (file.downloadHref) {
        if (!download) {
          download = document.createElement("a");
          download.textContent = "Download file";
          download.setAttribute("download", "");
          path.appendChild(download);
        }
        download.href = file.downloadHref;
      } else if (download) {
        download.remove();
      }

      panel.querySelector(".code-label").textContent = file.name;
      panel.querySelector(".code-language").textContent = file.language;
      const pre = panel.querySelector("pre");
      pre.dataset.language = file.language;
      const code = pre.querySelector("code");
      code.replaceChildren(...file.code.split("\n").map((line, lineIndex) => {
        const row = document.createElement("span");
        row.className = "code-line";
        const number = document.createElement("span");
        number.className = "line-number";
        number.setAttribute("aria-hidden", "true");
        number.textContent = String(lineIndex + 1).padStart(2, "0");
        const source = document.createElement("span");
        source.className = "line-source";
        source.textContent = line || " ";
        row.append(number, source);
        return row;
      }));

      const notes = panel.querySelector(".project-file-notes");
      notes.querySelector("h2").textContent = file.explanationTitle;
      notes.querySelectorAll("p:not(.eyebrow)").forEach((paragraph) => paragraph.remove());
      file.explanation.forEach((text) => {
        const paragraph = document.createElement("p");
        paragraph.textContent = text;
        notes.appendChild(paragraph);
      });
    };

    const selectFile = (index, updateHash = true) => {
      if (!Number.isInteger(index) || index < 0 || index >= files.length) return;
      files.forEach((file, fileIndex) => {
        const selected = fileIndex === index;
        file.classList.toggle("is-active", selected);
        file.setAttribute("aria-selected", String(selected));
      });
      renderFile(projectFiles[index], index);
      if (updateHash) history.replaceState(null, "", `#file-${index + 1}`);
    };

    files.forEach((file, index) => file.addEventListener("click", () => selectFile(index)));
    const requested = Number.parseInt(location.hash.match(/^#file-(\d+)$/)?.[1] ?? "1", 10) - 1;
    selectFile(requested >= 0 && requested < files.length ? requested : 0, false);
  });

  const shapes = document.querySelectorAll(".ambient-shapes");
  if (!("IntersectionObserver" in window)) {
    shapes.forEach((shape) => shape.classList.add("is-seen"));
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-seen");
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.12 });
  shapes.forEach((shape) => observer.observe(shape));
})();
