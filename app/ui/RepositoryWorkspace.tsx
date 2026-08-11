"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { ExampleRepository, RepositoryTopic } from "../repositoryExamples";
import { CodeBlock } from "./CodeBlock";

const topicLabels: Record<RepositoryTopic, string> = {
  inference: "Inference",
  data: "Datasets & labels",
  training: "Training & resume",
  validation: "Validation",
  export: "Export",
  examples: "Runnable applications",
  models: "Models & configuration",
  checkpoints: "Checkpoint contracts",
  api: "Python API",
  transforms: "Results & transforms",
};

function fileId(index: number) {
  return `file-${index + 1}`;
}

function fileType(language: string) {
  if (language === "python") return "PY";
  if (language === "yaml") return "YML";
  if (language === "json") return "JSN";
  return language.slice(0, 3).toUpperCase();
}

export function RepositoryWorkspace({
  topic,
  repository,
}: {
  topic: RepositoryTopic;
  repository: ExampleRepository;
}) {
  const [activeIndex, setActiveIndex] = useState(0);
  const sourceLines = repository.files.reduce((total, file) => total + file.code.split("\n").length, 0);
  const averageLines = Math.round(sourceLines / repository.files.length);
  const activeFile = repository.files[activeIndex];

  useEffect(() => {
    const syncHash = () => {
      const match = window.location.hash.match(/^#file-(\d+)$/);
      if (!match) return;
      const requested = Number(match[1]) - 1;
      if (requested >= 0 && requested < repository.files.length) setActiveIndex(requested);
    };
    const frame = window.requestAnimationFrame(syncHash);
    window.addEventListener("hashchange", syncHash);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("hashchange", syncHash);
    };
  }, [repository.files.length]);

  function selectFile(index: number) {
    setActiveIndex(index);
    window.history.replaceState(null, "", `#${fileId(index)}`);
  }

  return (
    <main id="content" className={`project-page tier-${repository.tier}`} data-project-workspace>
      <header className="project-header">
        <div className="project-breadcrumb">
          <Link href={`/docs/${topic}`}>← Back to {topicLabels[topic]}</Link>
          <div>
            <span>docs / {topic} / tier-{repository.tier}</span>
            {repository.downloadHref ? <a className="project-download" href={repository.downloadHref} download>Download project .zip</a> : null}
          </div>
        </div>
        <div className="project-heading">
          <div>
            <p className="eyebrow">Tier {repository.tier} · {repository.tier === 1 ? "Learning stage" : repository.tier === 2 ? "Usable example" : "System project"}</p>
            <h1>{repository.title}</h1>
            <p>{repository.summary}</p>
          </div>
          <dl>
            <div><dt>Mode</dt><dd>Read only</dd></div>
            <div><dt>Files</dt><dd>{repository.files.length}</dd></div>
            <div><dt>Source</dt><dd>{sourceLines.toLocaleString("en-US")} lines</dd></div>
            <div><dt>Average</dt><dd>{averageLines} / file</dd></div>
            <div><dt>Topic</dt><dd>{topicLabels[topic]}</dd></div>
          </dl>
        </div>
      </header>

      <div className="project-workbench">
        <aside className="project-explorer" aria-label={`${repository.title} file explorer`}>
          <div className="project-explorer-title"><span>Explorer</span><small>{repository.title}</small></div>
          <div className="project-file-list" role="tablist" aria-label="Project files">
            {repository.files.map((file, index) => (
              <button
                aria-controls="project-file-panel"
                aria-selected={activeIndex === index}
                className={activeIndex === index ? "is-active" : undefined}
                data-project-file={index}
                key={file.name}
                onClick={() => selectFile(index)}
                role="tab"
                type="button"
              >
                <span>{fileType(file.language)}</span>
                <b>{file.name}</b>
              </button>
            ))}
          </div>
        </aside>

        <section className="project-editor" aria-live="polite">
          <article
            aria-labelledby="project-file-label"
            data-project-panel={activeIndex}
            id="project-file-panel"
            key={activeFile.name}
            role="tabpanel"
          >
            <div className="project-editor-path">
              <span>fotonet-example</span><i>/</i><strong id="project-file-label">{activeFile.name}</strong>
              {activeFile.downloadHref ? <a href={activeFile.downloadHref} download>Download file</a> : null}
            </div>
            <div className="project-code-viewport">
              <CodeBlock code={activeFile.code} language={activeFile.language} label={activeFile.name} />
            </div>
            <section className="project-file-notes">
              <p className="eyebrow">File notes</p>
              <h2>{activeFile.explanationTitle}</h2>
              {activeFile.explanation.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
            </section>
          </article>
        </section>
      </div>
      <script
        data-project-files
        dangerouslySetInnerHTML={{ __html: JSON.stringify(repository.files).replaceAll("<", "\\u003c") }}
        type="application/json"
      />
    </main>
  );
}
