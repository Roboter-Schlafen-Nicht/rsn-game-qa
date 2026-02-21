import type { Severity } from "@/types";
import { severityBgColor } from "@/lib/utils";

interface SeverityBadgeProps {
  severity: Severity;
  count?: number;
}

export function SeverityBadge({ severity, count }: SeverityBadgeProps) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${severityBgColor(severity)}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${severity === "critical" ? "bg-red-400" : severity === "warning" ? "bg-amber-400" : "bg-blue-400"}`} />
      {severity}
      {count !== undefined && <span className="font-mono">({count})</span>}
    </span>
  );
}
