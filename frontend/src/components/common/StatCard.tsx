import React from "react";

interface StatCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  badge?: string;
  badgeVariant?: "blue" | "emerald" | "amber" | "rose" | "slate" | "purple";
  icon?: React.ReactNode;
}

export function StatCard({
  label,
  value,
  subValue,
  badge,
  badgeVariant = "slate",
  icon,
}: StatCardProps) {
  const badgeClasses = {
    blue: "bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-500/20",
    emerald: "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/20",
    amber: "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/20",
    rose: "bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/20",
    purple: "bg-purple-50 dark:bg-purple-500/10 text-purple-700 dark:text-purple-400 border-purple-200 dark:border-purple-500/20",
    slate: "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700",
  }[badgeVariant];

  return (
    <div className="bg-white dark:bg-[#0E131E] border border-slate-200 dark:border-[#1E293B] hover:border-blue-400 dark:hover:border-slate-600 transition-all rounded-xl p-4 flex flex-col justify-between shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] uppercase tracking-wider text-slate-500 dark:text-slate-400 font-semibold font-mono-num">
          {label}
        </span>
        {icon && <div className="text-slate-400 dark:text-slate-500">{icon}</div>}
      </div>

      <div className="flex items-baseline justify-between gap-2 mt-1">
        <div className="text-2xl font-bold font-mono-num text-slate-900 dark:text-slate-100 tracking-tight">
          {value}
        </div>
        {badge && (
          <span
            className={`text-[10px] font-mono-num font-semibold px-2 py-0.5 rounded border ${badgeClasses}`}
          >
            {badge}
          </span>
        )}
      </div>

      {subValue && (
        <div className="text-xs text-slate-500 dark:text-slate-400 mt-2 font-mono-num truncate">
          {subValue}
        </div>
      )}
    </div>
  );
}
