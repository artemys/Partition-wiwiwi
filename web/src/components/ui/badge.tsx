"use client";

import type { HTMLAttributes } from "react";

type Variant = "neutral" | "success" | "warning" | "danger";

const styles: Record<Variant, string> = {
  neutral: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200",
  success: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-200",
  warning: "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-200",
  danger: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-200",
};

export function Badge({
  variant = "neutral",
  className = "",
  ...props
}: HTMLAttributes<HTMLSpanElement> & { variant?: Variant }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-medium ${styles[variant]} ${className}`} {...props} />
  );
}
