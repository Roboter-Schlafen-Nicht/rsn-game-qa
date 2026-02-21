import { useState } from "react";
import {
  ClipboardCheck,
  Clock,
  Trophy,
  AlertTriangle,
  Bug,
} from "lucide-react";
import { StatCard } from "@/components/StatCard";
import { EpisodeChart } from "@/components/EpisodeChart";
import { FindingsTable } from "@/components/FindingsTable";
import { GameBadge } from "@/components/GameBadge";
import { MOCK_SESSIONS } from "@/data/mock";
import { formatNumber, gameDisplayName } from "@/lib/utils";

export function EvaluationPage() {
  const [selectedSession, setSelectedSession] = useState(0);
  const session = MOCK_SESSIONS[selectedSession];

  // Collect all findings from all episodes
  const allFindings = session.episodes.flatMap((ep) => ep.findings);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Evaluation</h1>
          <p className="text-sm text-zinc-500">
            Session evaluation results and episode analysis
          </p>
        </div>

        {/* Session selector */}
        <div className="flex gap-2">
          {MOCK_SESSIONS.map((s, i) => (
            <button
              key={s.session_id}
              onClick={() => setSelectedSession(i)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                i === selectedSession
                  ? "bg-zinc-800 text-zinc-100"
                  : "text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-300"
              }`}
            >
              {gameDisplayName(s.game)} - {s.build_id.includes("random") ? "Random" : "Trained"}
            </button>
          ))}
        </div>
      </div>

      {/* Session info */}
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/30 px-4 py-2.5 text-xs text-zinc-400">
        <GameBadge game={session.game} />
        <span className="text-zinc-600">|</span>
        <span>Model: <span className="text-zinc-300">{session.build_id}</span></span>
        <span className="text-zinc-600">|</span>
        <span>Session: <span className="font-mono text-zinc-300">{session.session_id}</span></span>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <StatCard
          label="Episodes"
          value={session.summary.total_episodes}
          icon={ClipboardCheck}
        />
        <StatCard
          label="Mean Length"
          value={session.summary.mean_episode_length.toFixed(1)}
          icon={Clock}
        />
        <StatCard
          label="Mean Reward"
          value={session.summary.mean_episode_reward.toFixed(2)}
          icon={Trophy}
        />
        <StatCard
          label="Critical Findings"
          value={session.summary.critical_findings}
          icon={AlertTriangle}
          color={session.summary.critical_findings > 0 ? "text-red-400" : "text-emerald-400"}
        />
        <StatCard
          label="Total Findings"
          value={formatNumber(session.summary.total_findings)}
          icon={Bug}
        />
      </div>

      {/* Episode chart */}
      <EpisodeChart episodes={session.episodes} />

      {/* Episodes table */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
        <div className="border-b border-zinc-800 px-5 py-3">
          <h2 className="text-sm font-medium text-zinc-300">Episodes</h2>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-left text-xs font-medium text-zinc-500">
              <th className="px-5 py-3">Episode</th>
              <th className="px-5 py-3">Steps</th>
              <th className="px-5 py-3">Reward</th>
              <th className="px-5 py-3">Status</th>
              <th className="px-5 py-3">FPS</th>
              <th className="px-5 py-3">Duration</th>
              <th className="px-5 py-3">Findings</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {session.episodes.map((ep) => (
              <tr key={ep.episode_id} className="hover:bg-zinc-800/30 transition-colors">
                <td className="px-5 py-2.5 font-mono text-xs text-zinc-300">
                  E{ep.episode_id}
                </td>
                <td className="px-5 py-2.5 font-mono text-xs text-zinc-400">
                  {formatNumber(ep.steps)}
                </td>
                <td className={`px-5 py-2.5 font-mono text-xs ${ep.total_reward >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {ep.total_reward.toFixed(2)}
                </td>
                <td className="px-5 py-2.5">
                  <span className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-medium ${
                    ep.truncated
                      ? "bg-amber-500/10 text-amber-400"
                      : ep.steps < 10
                        ? "bg-red-500/10 text-red-400"
                        : "bg-zinc-800 text-zinc-400"
                  }`}>
                    {ep.truncated ? "truncated" : ep.terminated ? "terminated" : "running"}
                  </span>
                </td>
                <td className="px-5 py-2.5 font-mono text-xs text-zinc-400">
                  {ep.metrics.mean_fps?.toFixed(1) ?? "-"}
                </td>
                <td className="px-5 py-2.5 font-mono text-xs text-zinc-400">
                  {ep.metrics.total_duration_seconds != null
                    ? ep.metrics.total_duration_seconds < 1
                      ? `${(ep.metrics.total_duration_seconds * 1000).toFixed(0)}ms`
                      : `${ep.metrics.total_duration_seconds.toFixed(1)}s`
                    : "-"}
                </td>
                <td className="px-5 py-2.5 font-mono text-xs text-zinc-400">
                  {ep.findings.length || "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Findings from this session */}
      {allFindings.length > 0 && (
        <div>
          <h2 className="mb-3 text-sm font-medium text-zinc-300">Session Findings</h2>
          <FindingsTable findings={allFindings} />
        </div>
      )}
    </div>
  );
}
