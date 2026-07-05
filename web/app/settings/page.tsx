"use client";

import { useState } from "react";
import { ErrorMessage } from "@/components/ErrorMessage";
import { API_BASE, saveState } from "@/lib/api";

export default function SettingsPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  async function handleSave() {
    setError(null);
    setSuccess(null);
    setLoading(true);
    try {
      const res = await saveState();
      setSuccess(`${res.message} Files: ${res.files.join(", ")}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="mt-1 text-velox-muted">
          Persist vectors, IVF index, and document metadata to disk.
        </p>
      </div>

      <div className="rounded-lg border border-velox-border bg-velox-card p-4">
        <p className="text-sm text-velox-muted">API base URL</p>
        <p className="mt-1 font-mono text-sm text-white">{API_BASE}</p>
        <p className="mt-2 text-xs text-velox-muted">
          Set NEXT_PUBLIC_API_URL in web/.env.local to override.
        </p>
      </div>

      <ErrorMessage message={error} />
      {success && (
        <p className="rounded-md border border-emerald-800 bg-emerald-950/40 px-3 py-2 text-sm text-emerald-300">
          {success}
        </p>
      )}

      <button
        type="button"
        onClick={handleSave}
        disabled={loading}
        className="rounded-md bg-velox-accent px-4 py-2 text-sm font-medium text-white hover:bg-blue-600 disabled:opacity-50"
      >
        {loading ? "Saving…" : "Save database to disk"}
      </button>
    </div>
  );
}
