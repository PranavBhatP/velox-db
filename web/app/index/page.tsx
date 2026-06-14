"use client";

import { useEffect, useState } from "react";
import { ErrorMessage } from "@/components/ErrorMessage";
import { getHealth, trainIndex } from "@/lib/api";

export default function TrainIndexPage() {
  const [numClusters, setNumClusters] = useState(10);
  const [epochs, setEpochs] = useState(15);
  const [metric, setMetric] = useState("eucl");
  const [vectorCount, setVectorCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then((h) => {
        setVectorCount(h.vector_count);
        const suggested = Math.max(1, Math.floor(Math.sqrt(h.vector_count)));
        if (h.vector_count > 0) setNumClusters(suggested);
      })
      .catch(() => {});
  }, []);

  async function handleTrain(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setLoading(true);
    try {
      const res = await trainIndex(numClusters, epochs, metric);
      setSuccess(res.message);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed");
    } finally {
      setLoading(false);
    }
  }

  const hint =
    vectorCount > 0
      ? `Suggested clusters ≈ √${vectorCount} = ${Math.max(1, Math.floor(Math.sqrt(vectorCount)))}`
      : "Add documents before training";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Train IVF index</h1>
        <p className="mt-1 text-velox-muted">
          Build a K-means inverted file index for faster approximate search.
        </p>
      </div>

      <p className="text-sm text-velox-muted">{hint}</p>

      <ErrorMessage message={error} />
      {success && (
        <p className="rounded-md border border-emerald-800 bg-emerald-950/40 px-3 py-2 text-sm text-emerald-300">
          {success}
        </p>
      )}

      <form onSubmit={handleTrain} className="max-w-md space-y-4">
        <Field
          label="Number of clusters"
          type="number"
          min={1}
          value={numClusters}
          onChange={(v) => setNumClusters(Number(v))}
        />
        <Field
          label="Epochs"
          type="number"
          min={1}
          value={epochs}
          onChange={(v) => setEpochs(Number(v))}
        />
        <div>
          <label className="text-sm text-velox-muted">Metric</label>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            className="mt-1 block w-full rounded-md border border-velox-border bg-velox-card px-2 py-2 text-white"
          >
            <option value="eucl">Euclidean</option>
            <option value="cos">Cosine</option>
          </select>
        </div>
        <button
          type="submit"
          disabled={loading || vectorCount === 0}
          className="rounded-md bg-velox-accent px-4 py-2 text-sm font-medium text-white hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? "Training…" : "Train index"}
        </button>
      </form>
    </div>
  );
}

function Field({
  label,
  type,
  min,
  value,
  onChange,
}: {
  label: string;
  type: string;
  min: number;
  value: number;
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="text-sm text-velox-muted">{label}</label>
      <input
        type={type}
        min={min}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 block w-full rounded-md border border-velox-border bg-velox-card px-2 py-2 text-white"
      />
    </div>
  );
}
