import type { FindingReport } from "@/types";
import { SeverityBadge } from "@/components/SeverityBadge";

interface FindingsTableProps {
  findings: FindingReport[];
  maxRows?: number;
}

export function FindingsTable({ findings, maxRows }: FindingsTableProps) {
  const sorted = [...findings].sort((a, b) => {
    const order: Record<string, number> = { critical: 0, warning: 1, info: 2 };
    return (order[a.severity] ?? 3) - (order[b.severity] ?? 3) || a.step - b.step;
  });
  const displayed = maxRows ? sorted.slice(0, maxRows) : sorted;

  if (displayed.length === 0) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-8 text-center text-sm text-zinc-500">
        No findings recorded
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 text-left text-xs font-medium text-zinc-500">
            <th className="px-4 py-3">Severity</th>
            <th className="px-4 py-3">Oracle</th>
            <th className="px-4 py-3">Step</th>
            <th className="px-4 py-3">Description</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800/50">
          {displayed.map((f, i) => (
            <tr key={i} className="hover:bg-zinc-800/30 transition-colors">
              <td className="px-4 py-2.5">
                <SeverityBadge severity={f.severity} />
              </td>
              <td className="px-4 py-2.5 font-mono text-xs text-zinc-300">
                {f.oracle_name}
              </td>
              <td className="px-4 py-2.5 font-mono text-xs text-zinc-400">
                {f.step}
              </td>
              <td className="px-4 py-2.5 text-xs text-zinc-400 max-w-md truncate">
                {f.description}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {maxRows && sorted.length > maxRows && (
        <div className="border-t border-zinc-800 px-4 py-2 text-center text-xs text-zinc-500">
          Showing {maxRows} of {sorted.length} findings
        </div>
      )}
    </div>
  );
}
