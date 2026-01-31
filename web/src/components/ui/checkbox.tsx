import { type InputHTMLAttributes, type ReactNode } from "react";

type CheckboxProps = InputHTMLAttributes<HTMLInputElement> & {
  children?: ReactNode;
};

export function Checkbox({ className = "", children, ...props }: CheckboxProps) {
  return (
    <label
      className={`flex items-center gap-2 text-sm font-medium text-zinc-900 dark:text-zinc-100 ${className}`}
    >
      <input
        type="checkbox"
        className="h-4 w-4 rounded border-zinc-300 text-indigo-500 shadow-sm focus:ring-indigo-500 dark:border-zinc-600"
        {...props}
      />
      {children}
    </label>
  );
}
