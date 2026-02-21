import { AlertTriangle, Bug, Info, Shield } from "lucide-react";
import { StatCard } from "@/components/StatCard";
import { SeverityBadge } from "@/components/SeverityBadge";
import { GameBadge } from "@/components/GameBadge";
import { MOCK_SESSIONS } from "@/data/mock";
import { ORACLE_INFO } from "@/types";
import { formatNumber } from "@/lib/utils";

export function FindingsPage() {
  // Aggregate all findings across all sessions
  const allFindings = MOCK_SESSIONS.flatMap((s) =>
    s.episodes.flatMap((ep) =>
      ep.findings.map((f) => ({
        ...f,
        game: s.game,
        build: s.build_id,
        episode: ep.episode_id,
        session: s.session_id,
      })),
    ),
  );

  const critical = allFindings.filter((f) => f.severity === "critical");
  const warnings = allFindings.filter((f) => f.severity === "warning");
  const info = allFindings.filter((f) => f.severity === "info");

  // Group by oracle
  const byOracle = allFindings.reduce<Record<string, typeof allFindings>>((acc, f) => {
    if (!acc[f.oracle_name]) acc[f.oracle_name] = [];
    acc[f.oracle_name].push(f);
    return acc;
  }, {});

  // Summary stats from session summaries (includes all findings, not just the detailed ones)
  const totalFromSummaries = MOCK_SESSIONS.reduce((sum, s) => sum + s.summary.total_findings, 0);
  const criticalFromSummaries = MOCK_SESSIONS.reduce((sum, s) => sum + s.summary.critical_findings, 0);
  const warningFromSummaries = MOCK_SESSIONS.reduce((sum, s) => sum + s.summary.warning_findings, 0);
  const infoFromSummaries = MOCK_SESSIONS.reduce((sum, s) => sum + s.summary.info_findings, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Findings</h1>
        <p className="text-sm text-zinc-500">
          Oracle findings across all evaluation sessions
        </p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          label="Total Findings"
          value={formatNumber(totalFromSummaries)}
          icon={Bug}
        />
        <StatCard
          label="Critical"
          value={formatNumber(criticalFromSummaries)}
          icon={AlertTriangle}
          color="text-red-400"
        />
        <StatCard
          label="Warnings"
          value={formatNumber(warningFromSummaries)}
          icon={Shield}
          color="text-amber-400"
        />
        <StatCard
          label="Info"
          value={formatNumber(infoFromSummaries)}
          icon={Info}
          color="text-blue-400"
        />
      </div>

      {/* Findings by oracle */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
        <div className="border-b border-zinc-800 px-5 py-3">
          <h2 className="text-sm font-medium text-zinc-300">Findings by Oracle</h2>
        </div>
        <div className="divide-y divide-zinc-800/50">
          {Object.entries(byOracle)
            .sort(([, a], [, b]) => b.length - a.length)
            .map(([oracleName, findings]) => {
              const oracleInfo = ORACLE_INFO[oracleName as keyof typeof ORACLE_INFO];
              return (
                <div key={oracleName} className="px-5 py-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium text-zinc-200">
                        {oracleInfo?.label ?? oracleName}
                      </span>
                      <span className="text-xs text-zinc-500">
                        {oracleInfo?.description ?? ""}
                      </span>
                    </div>
                    <span className="font-mono text-sm text-zinc-400">
                      {findings.length}
                    </span>
                  </div>
                  <div className="mt-2 space-y-1.5">
                    {findings.map((f, i) => (
                      <div key={i} className="flex items-center gap-3 text-xs">
                        <SeverityBadge severity={f.severity} />
                        <GameBadge game={f.game} />
                        <span className="font-mono text-zinc-500">
                          {f.build} &middot; E{f.episode} &middot; Step {f.step}
                        </span>
                        <span className="text-zinc-400 truncate max-w-xs">
                          {f.description}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Critical findings detail */}
      {critical.length > 0 && (
        <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-5">
          <h2 className="mb-3 text-sm font-medium text-red-400">
            Critical Findings Detail ({critical.length})
          </h2>
          <div className="space-y-3">
            {critical.map((f, i) => (
              <div key={i} className="rounded-lg bg-zinc-900/60 p-4">
                <div className="flex items-center gap-3 mb-2">
                  <SeverityBadge severity="critical" />
                  <GameBadge game={f.game} />
                  <span className="font-mono text-xs text-zinc-500">
                    {f.build} &middot; Episode {f.episode} &middot; Step {f.step}
                  </span>
                </div>
                <p className="text-sm text-zinc-300">{f.description}</p>
                {Object.keys(f.data).length > 0 && (
                  <pre className="mt-2 rounded-md bg-zinc-950 p-2 text-[11px] text-zinc-500 font-mono overflow-x-auto">
                    {JSON.stringify(f.data, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-5">
          <h2 className="mb-3 text-sm font-medium text-amber-400">
            Warnings ({warnings.length})
          </h2>
          <div className="space-y-2">
            {warnings.map((f, i) => (
              <div key={i} className="flex items-center gap-3 text-xs">
                <GameBadge game={f.game} />
                <span className="font-mono text-zinc-500">
                  E{f.episode} Step {f.step}
                </span>
                <span className="text-zinc-400">{f.description}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info */}
      {info.length > 0 && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
          <h2 className="mb-3 text-sm font-medium text-zinc-400">
            Info ({info.length})
          </h2>
          <p className="text-xs text-zinc-500">
            {info.length} informational findings across all sessions. These are typically
            episode length notifications, temporal observations, and reward consistency checks.
          </p>
        </div>
      )}
    </div>
  );
}
