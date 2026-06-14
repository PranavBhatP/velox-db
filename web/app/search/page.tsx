"use client";

import { useState } from "react";
import { ErrorMessage } from "@/components/ErrorMessage";
import { StatusBadge } from "@/components/StatusBadge";
import { searchByText, type SearchResult } from "@/lib/api";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [metric, setMetric] = useState("eucl");
  const [k, setK] = useState(5);
  const [nprobe, setNprobe] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [indexed, setIndexed] = useState<boolean | null>(null);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResults([]);
    setLoading(true);
    try {
      const res = await searchByText(query.trim(), metric, k, nprobe);
      setResults(res.results);
      setIndexed(res.is_indexed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Search</h1>
        <p className="mt-1 text-velox-muted">
          Query text is embedded and matched against stored vectors.
        </p>
      </div>

      <form onSubmit={handleSearch} className="space-y-4">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={4}
          placeholder="Enter search query…"
          className="w-full rounded-md border border-velox-border bg-velox-card px-3 py-2 text-white"
          required
        />

        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-velox-muted">Metric</label>
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
              className="rounded-md border border-velox-border bg-velox-card px-2 py-1 text-sm text-white"
            >
              <option value="eucl">Euclidean</option>
              <option value="cos">Cosine</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm text-velox-muted">Top-k</label>
            <input
              type="number"
              min={1}
              max={100}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-16 rounded-md border border-velox-border bg-velox-card px-2 py-1 text-sm text-white"
            />
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm text-velox-muted">nprobe</label>
            <input
              type="number"
              min={1}
              max={100}
              value={nprobe}
              onChange={(e) => setNprobe(Number(e.target.value))}
              className="w-16 rounded-md border border-velox-border bg-velox-card px-2 py-1 text-sm text-white"
            />
            <span className="text-xs text-velox-muted">clusters to probe</span>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-velox-accent px-4 py-2 text-sm font-medium text-white hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      <ErrorMessage message={error} />

      {indexed !== null && (
        <div>
          <StatusBadge
            ok={indexed}
            label={indexed ? "IVF index active" : "Brute-force (index not trained)"}
          />
        </div>
      )}

      {results.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-medium text-white">
            Top {results.length} result{results.length !== 1 ? "s" : ""}
          </h2>
          {results.map((r, rank) => (
            <div
              key={r.id}
              className="rounded-lg border border-velox-border bg-velox-card p-4"
            >
              <div className="flex items-center justify-between">
                <p className="text-xs text-velox-muted">
                  #{rank + 1} · ID {r.id}
                </p>
                <p className="text-xs text-velox-muted">
                  dist {r.distance.toFixed(4)}
                </p>
              </div>
              <p className="mt-2 whitespace-pre-wrap text-slate-200">
                {r.text ?? "(no metadata for this vector)"}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
