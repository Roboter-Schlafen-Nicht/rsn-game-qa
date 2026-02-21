import {
  Bug,
  Gamepad2,
  AlertTriangle,
  Eye,
  Clock,
} from "lucide-react";
import { StatCard } from "@/components/StatCard";
import { GameBadge } from "@/components/GameBadge";
import { SeverityBadge } from "@/components/SeverityBadge";
import { MOCK_SESSIONS, MOCK_TRAINING_ANALYSES } from "@/data/mock";
import { formatNumber } from "@/lib/utils";

export function OverviewPage() {
  // Aggregate stats across all sessions
  const totalFindings = MOCK_SESSIONS.reduce((sum, s) => sum + s.summary.total_findings, 0);
  const criticalFindings = MOCK_SESSIONS.reduce((sum, s) => sum + s.summary.critical_findings, 0);
  const totalEpisodes = MOCK_SESSIONS.reduce((sum, s) => sum + s.summary.total_episodes, 0);
  const games = [...new Set(MOCK_SESSIONS.map((s) => s.game))];
  const totalUniqueStates = MOCK_TRAINING_ANALYSES.reduce(
    (sum, a) => sum + a.coverage.final_unique_states,
    0,
  );

  // Recent critical findings across all sessions
  const recentCritical = MOCK_SESSIONS.flatMap((s) =>
    s.episodes.flatMap((ep) =>
      ep.findings
        .filter((f) => f.severity === "critical")
        .map((f) => ({ ...f, game: s.game, build: s.build_id, episode: ep.episode_id })),
    ),
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-sm text-zinc-500">
          RL-driven autonomous game testing platform
        </p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <StatCard
          label="Games Tested"
          value={games.length}
          icon={Gamepad2}
          subtitle={games.map((g) => g).join(", ")}
        />
        <StatCard
          label="Total Episodes"
          value={formatNumber(totalEpisodes)}
          icon={Clock}
        />
        <StatCard
          label="Total Findings"
          value={formatNumber(totalFindings)}
          icon={Bug}
        />
        <StatCard
          label="Critical Bugs"
          value={criticalFindings}
          icon={AlertTriangle}
          color="text-red-400"
        />
        <StatCard
          label="Unique States"
          value={formatNumber(totalUniqueStates)}
          icon={Eye}
          subtitle="Across all training"
        />
      </div>

      {/* Sessions overview */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
        <div className="border-b border-zinc-800 px-5 py-3">
          <h2 className="text-sm font-medium text-zinc-300">Evaluation Sessions</h2>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-left text-xs font-medium text-zinc-500">
              <th className="px-5 py-3">Game</th>
              <th className="px-5 py-3">Model</th>
              <th className="px-5 py-3">Episodes</th>
              <th className="px-5 py-3">Mean Length</th>
              <th className="px-5 py-3">Mean Reward</th>
              <th className="px-5 py-3">Findings</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {MOCK_SESSIONS.map((s) => (
              <tr key={s.session_id} className="hover:bg-zinc-800/30 transition-colors">
                <td className="px-5 py-3">
                  <GameBadge game={s.game} />
                </td>
                <td className="px-5 py-3 font-mono text-xs text-zinc-300">
                  {s.build_id}
                </td>
                <td className="px-5 py-3 font-mono text-xs text-zinc-400">
                  {s.summary.total_episodes}
                </td>
                <td className="px-5 py-3 font-mono text-xs text-zinc-400">
                  {s.summary.mean_episode_length.toFixed(1)}
                </td>
                <td className="px-5 py-3 font-mono text-xs text-zinc-400">
                  {s.summary.mean_episode_reward.toFixed(2)}
                </td>
                <td className="px-5 py-3">
                  <div className="flex gap-2">
                    {s.summary.critical_findings > 0 && (
                      <SeverityBadge severity="critical" count={s.summary.critical_findings} />
                    )}
                    {s.summary.warning_findings > 0 && (
                      <SeverityBadge severity="warning" count={s.summary.warning_findings} />
                    )}
                    <SeverityBadge severity="info" count={s.summary.info_findings} />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Critical findings */}
      {recentCritical.length > 0 && (
        <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-5">
          <h2 className="mb-3 text-sm font-medium text-red-400">Critical Findings</h2>
          <div className="space-y-2">
            {recentCritical.map((f, i) => (
              <div key={i} className="flex items-start gap-3 rounded-lg bg-zinc-900/50 px-4 py-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <GameBadge game={f.game} />
                    <span className="font-mono text-xs text-zinc-500">
                      {f.build} &middot; E{f.episode} &middot; Step {f.step}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-zinc-400">{f.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
