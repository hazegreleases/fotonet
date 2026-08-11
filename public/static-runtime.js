(() => {
  document.querySelectorAll(".code-block").forEach((block) => {
    const button = block.querySelector("figcaption button");
    const lines = [...block.querySelectorAll(".line-source")];
    if (!button || lines.length === 0) return;
    const source = lines.map((line) => line.textContent ?? "").join("\n");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(source);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Select code";
      }
      window.setTimeout(() => { button.textContent = "Copy"; }, 1800);
    });
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
