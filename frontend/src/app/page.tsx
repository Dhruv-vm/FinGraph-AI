import type { Metadata } from "next";
import { HomePage } from "@/components/home/HomePage";

export const metadata: Metadata = {
  title: "FinGraph AI — Understand finance beyond documents",
  description:
    "FinGraph AI connects SEC filings into a temporal knowledge graph and answers point-in-time questions with hybrid GraphRAG — grounded in cited evidence, or not at all.",
};

export default function Page() {
  return <HomePage />;
}
