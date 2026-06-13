"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ErrorMessage } from "@/components/ErrorMessage";
import { StatusBadge } from "@/components/StatusBadge";
import { getHealth, type HealthResponse } from "@/lib/api";

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  const hasVectors = (health?.vector_count ?? 0) > 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="mt-1 text-velox-muted">
          Semantic search over text stored as vectors in VeloxDB.
        </p>
      </div>

      <ErrorMessage message={error} />

      {loading && (
        <p className="text-sm text-velox-muted">Connecting to API…</p>
      )}

      {health && (
        <div className="grid gap-4 sm:grid-cols-2">
          <StatCard label="Vectors" value={String(health.vector_count)} />
          <StatCard
            label="Dimension"
            value={health.dim != null ? String(health.dim) : "—"}
          />
          <div className="rounded-lg border border-velox-border bg-velox-card p-4">
            <p className="text-sm text-velox-muted">IVF index</p>
            <div className="mt-2">
              <StatusBadge
                ok={health.is_indexed}
                label={health.is_indexed ? "Trained" : "Not trained"}
              />
            </div>
          </div>
          <StatCard
            label="Embedding dim"
            value={String(health.expected_embedding_dim)}
          />
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        <ActionLink href="/ingest" disabled={false}>
          Add documents
        </ActionLink>
        <ActionLink href="/search" disabled={!hasVectors}>
          Search
        </ActionLink>
        <ActionLink href="/index" disabled={!hasVectors}>
          Train index
        </ActionLink>
        <ActionLink href="/data" disabled={!hasVectors}>
          View documents
        </ActionLink>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-velox-border bg-velox-card p-4">
      <p className="text-sm text-velox-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-white">{value}</p>
    </div>
  );
}

function ActionLink({
  href,
  children,
  disabled,
}: {
  href: string;
  children: React.ReactNode;
  disabled: boolean;
}) {
  if (disabled) {
    return (
      <span className="cursor-not-allowed rounded-md bg-slate-800 px-4 py-2 text-sm text-slate-500">
        {children}
      </span>
    );
  }
  return (
    <Link
      href={href}
      className="rounded-md bg-velox-accent px-4 py-2 text-sm font-medium text-white hover:bg-blue-600"
    >
      {children}
    </Link>
  );
}
