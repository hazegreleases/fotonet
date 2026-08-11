import { notFound } from "next/navigation";
import { pageMetadata } from "../../../seo";
import {
  repositoryExamples,
  type RepositoryTopic,
} from "../../../repositoryExamples";
import { RepositoryWorkspace } from "../../../ui/RepositoryWorkspace";

const topics = Object.keys(repositoryExamples) as RepositoryTopic[];

function resolveProject(topicValue: string, tierValue: string) {
  if (!topics.includes(topicValue as RepositoryTopic)) return null;
  const topic = topicValue as RepositoryTopic;
  const match = tierValue.match(/^tier-([123])$/);
  if (!match) return null;
  const tier = Number(match[1]);
  const repository = repositoryExamples[topic].find((item) => item.tier === tier);
  return repository ? { topic, repository } : null;
}

export function generateStaticParams() {
  return topics.flatMap((topic) => [1, 2, 3].map((tier) => ({ topic, tier: `tier-${tier}` })));
}

export async function generateMetadata({ params }: { params: Promise<{ topic: string; tier: string }> }) {
  const values = await params;
  const project = resolveProject(values.topic, values.tier);
  if (!project) return {};
  return pageMetadata({
    title: `${project.repository.title} · Tier ${project.repository.tier}`,
    description: `${project.repository.summary} Browse every file in a read-only workspace with implementation notes.`,
    path: `/code/${project.topic}/tier-${project.repository.tier}`,
    keywords: ["fotonet example", "object detection code", `${project.topic} example`],
  });
}

export default async function ProjectPage({ params }: { params: Promise<{ topic: string; tier: string }> }) {
  const values = await params;
  const project = resolveProject(values.topic, values.tier);
  if (!project) notFound();
  return <RepositoryWorkspace topic={project.topic} repository={project.repository} />;
}
