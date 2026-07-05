export function StatusBadge({
  ok,
  label,
}: {
  ok: boolean;
  label: string;
}) {
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
        ok
          ? "bg-emerald-900/50 text-emerald-300"
          : "bg-amber-900/50 text-amber-300"
      }`}
    >
      {label}
    </span>
  );
}
