"use client";

import { useState } from "react";
import { ErrorMessage } from "@/components/ErrorMessage";
import { addDocument, addDocumentsBatch } from "@/lib/api";

export default function IngestPage() {
  const [text, setText] = useState("");
  const [bulk, setBulk] = useState("");
  const [mode, setMode] = useState<"single" | "bulk">("single");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setLoading(true);
    try {
      if (mode === "single") {
        const res = await addDocument(text.trim());
        setSuccess(`Document added with ID ${res.id}`);
        setText("");
      } else {
        const lines = bulk
          .split("\n")
          .map((l) => l.trim())
          .filter(Boolean);
        if (lines.length === 0) {
          throw new Error("Add at least one non-empty line");
        }
        const res = await addDocumentsBatch(lines);
        setSuccess(`Added ${res.count} document(s): IDs ${res.ids.join(", ")}`);
        setBulk("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ingest failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Ingest documents</h1>
        <p className="mt-1 text-velox-muted">
          Text is embedded locally with sentence-transformers (all-MiniLM-L6-v2)
          and stored in VeloxDB.
        </p>
      </div>

      <div className="flex gap-2">
        <ModeButton active={mode === "single"} onClick={() => setMode("single")}>
          Single
        </ModeButton>
        <ModeButton active={mode === "bulk"} onClick={() => setMode("bulk")}>
          Bulk (one line per document)
        </ModeButton>
      </div>

      <ErrorMessage message={error} />
      {success && (
        <p className="rounded-md border border-emerald-800 bg-emerald-950/40 px-3 py-2 text-sm text-emerald-300">
          {success}
        </p>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === "single" ? (
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={6}
            placeholder="Enter document text…"
            className="w-full rounded-md border border-velox-border bg-velox-card px-3 py-2 text-white placeholder:text-slate-500"
            required
          />
        ) : (
          <textarea
            value={bulk}
            onChange={(e) => setBulk(e.target.value)}
            rows={10}
            placeholder={"Line one\nLine two\n…"}
            className="w-full rounded-md border border-velox-border bg-velox-card px-3 py-2 font-mono text-sm text-white placeholder:text-slate-500"
            required
          />
        )}

        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-velox-accent px-4 py-2 text-sm font-medium text-white hover:bg-blue-600 disabled:opacity-50"
        >
          {loading
            ? "Embedding & storing… (first run may download the model)"
            : "Add to database"}
        </button>
      </form>
    </div>
  );
}

function ModeButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-md px-3 py-1.5 text-sm ${
        active
          ? "bg-velox-accent text-white"
          : "bg-velox-card text-velox-muted hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}
