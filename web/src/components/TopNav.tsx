"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/library", label: "Bibliothèque" },
  { href: "/new", label: "Nouvelle transcription" },
  { href: "/settings", label: "Paramètres" },
];

export function TopNav() {
  const pathname = usePathname();
  return (
    <nav className="border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
        <Link href="/library" className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
          TabScore
        </Link>
        <div className="flex flex-wrap items-center gap-2">
          {links.map((link) => {
            const active = pathname === link.href || pathname.startsWith(`${link.href}/`);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-full px-3 py-1 text-sm transition ${
                  active
                    ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                    : "text-zinc-600 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-900"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
