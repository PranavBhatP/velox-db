"use client";

import { useEffect, useState } from "react";
import { ErrorMessage } from "@/components/ErrorMessage";
import { listDocuments, type DocumentItem } from "@/lib/api";

export default function DataPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDocuments(0, 200)
      .then((res) => {
        setDocs(res.documents);
        setTotal(res.total);
      })
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Failed to load documents")
      )
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Documents</h1>
        <p className="mt-1 text-velox-muted">
          {total} document(s) with stored text metadata.
        </p>
      </div>

      <ErrorMessage message={error} />

      {loading && <p className="text-sm text-velox-muted">Loading…</p>}

      {!loading && docs.length === 0 && !error && (
        <p className="text-velox-muted">No documents yet. Add some from Ingest.</p>
      )}

      <div className="overflow-x-auto rounded-lg border border-velox-border">
        <table className="w-full text-left text-sm">
          <thead className="bg-velox-card text-velox-muted">
            <tr>
              <th className="px-4 py-2 font-medium">ID</th>
              <th className="px-4 py-2 font-medium">Text</th>
              <th className="px-4 py-2 font-medium">Created</th>
            </tr>
          </thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.id} className="border-t border-velox-border">
                <td className="px-4 py-3 align-top text-velox-muted">{d.id}</td>
                <td className="max-w-md px-4 py-3 align-top whitespace-pre-wrap">
                  {d.text}
                </td>
                <td className="px-4 py-3 align-top text-xs text-velox-muted">
                  {d.created_at
                    ? new Date(d.created_at).toLocaleString()
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
