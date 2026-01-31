"use client";

import type { HTMLAttributes } from "react";

export function Progress({
  value,
  className = "",
  ...props
}: HTMLAttributes<HTMLDivElement> & { value: number }) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div
      {...props}
      className={`h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800 ${className}`}
    >
      <div className="h-full bg-zinc-900 transition-all dark:bg-zinc-100" style={{ width: `${clamped}%` }} />
    </div>
  );
}
