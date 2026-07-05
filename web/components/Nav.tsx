import Link from "next/link";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/ingest", label: "Ingest" },
  { href: "/search", label: "Search" },
  { href: "/index", label: "Train index" },
  { href: "/data", label: "Documents" },
  { href: "/settings", label: "Settings" },
];

export function Nav() {
  return (
    <nav className="border-b border-velox-border bg-velox-card">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-4 px-4 py-3">
        <Link href="/" className="text-lg font-semibold text-white">
          VeloxDB
        </Link>
        <div className="flex flex-wrap gap-3 text-sm">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="text-velox-muted hover:text-white"
            >
              {l.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
