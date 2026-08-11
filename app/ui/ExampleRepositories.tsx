import Link from "next/link";
import { repositoryExamples, type RepositoryTopic } from "../repositoryExamples";

const tierLabels = {
  1: "Learning stage",
  2: "Usable example",
  3: "System project",
} as const;

const tierDescriptions = {
  1: "One concept, two files, and no infrastructure around it.",
  2: "A more complete project intended to be copied and adapted.",
  3: "An interconnected application that treats this topic as its primary system boundary.",
} as const;

export function ExampleRepositories({ topic }: { topic: RepositoryTopic }) {
  const repositories = repositoryExamples[topic];

  return (
    <section className="repository-section" aria-labelledby={`${topic}-repositories`}>
      <div className="repository-section-heading">
        <p className="eyebrow">Example projects</p>
        <h2 id={`${topic}-repositories`}>Choose the depth you need.</h2>
        <p>Each tier opens a separate read-only repository workspace with a file tree, one active file, and implementation notes.</p>
      </div>
      <div className="repository-launch-grid">
        {repositories.map((repository) => (
          <Link
            className={`repository-launch-card tier-${repository.tier}`}
            href={`/code/${topic}/tier-${repository.tier}`}
            key={repository.tier}
          >
            <span className="repository-launch-tier">Tier {repository.tier} · {tierLabels[repository.tier]}</span>
            <h3>{repository.title}</h3>
            <p>{tierDescriptions[repository.tier]}</p>
            <footer>
              <span>{repository.files.length} files / {repository.files.reduce((total, file) => total + file.code.split("\n").length, 0).toLocaleString("en-US")} lines</span>
              <strong>Open project →</strong>
            </footer>
          </Link>
        ))}
      </div>
    </section>
  );
}
