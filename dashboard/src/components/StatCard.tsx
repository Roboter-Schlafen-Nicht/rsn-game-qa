import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  icon: LucideIcon;
  subtitle?: string;
  trend?: { value: number; label: string };
  color?: string;
  children?: ReactNode;
}

export function StatCard({ label, value, icon: Icon, subtitle, trend, color = "text-zinc-100", children }: StatCardProps) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 backdrop-blur-sm">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-sm font-medium text-zinc-400">{label}</p>
          <p className={`text-2xl font-semibold tracking-tight ${color}`}>
            {value}
          </p>
          {subtitle && (
            <p className="text-xs text-zinc-500">{subtitle}</p>
          )}
          {trend && (
            <p className={`text-xs font-medium ${trend.value >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {trend.value >= 0 ? "+" : ""}{trend.value}% {trend.label}
            </p>
          )}
        </div>
        <div className="rounded-lg bg-zinc-800/50 p-2.5">
          <Icon className="h-5 w-5 text-zinc-400" />
        </div>
      </div>
      {children}
    </div>
  );
}
