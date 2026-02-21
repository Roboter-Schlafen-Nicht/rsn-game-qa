import { useState } from "react";
import {
  GraduationCap,
  Clock,
  Eye,
  Activity,
  Gauge,
} from "lucide-react";
import { StatCard } from "@/components/StatCard";
import { CoverageChart } from "@/components/CoverageChart";
import { GameBadge } from "@/components/GameBadge";
import { MOCK_TRAINING_ANALYSES } from "@/data/mock";
import { formatNumber, gameDisplayName, gameChartColor } from "@/lib/utils";

export function TrainingPage() {
  const [selectedGame, setSelectedGame] = useState(0);
  const analysis = MOCK_TRAINING_ANALYSES[selectedGame];
  const config = analysis.config;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Training</h1>
          <p className="text-sm text-zinc-500">
            Training run analysis and metrics
          </p>
        </div>

        {/* Game selector */}
        <div className="flex gap-2">
          {MOCK_TRAINING_ANALYSES.map((a, i) => (
            <button
              key={a.config.game}
              onClick={() => setSelectedGame(i)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                i === selectedGame
                  ? "bg-zinc-800 text-zinc-100"
                  : "text-zinc-500 hover:bg-zinc-800/50 hover:text-zinc-300"
              }`}
            >
              {gameDisplayName(a.config.game)}
            </button>
          ))}
        </div>
      </div>

      {/* Config summary */}
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/30 px-4 py-2.5 text-xs text-zinc-400">
        <GameBadge game={config.game} />
        <span className="text-zinc-600">|</span>
        <span>Policy: <span className="text-zinc-300">{config.policy.toUpperCase()}</span></span>
        <span className="text-zinc-600">|</span>
        <span>Reward: <span className="text-zinc-300">{config.reward_mode}</span></span>
        <span className="text-zinc-600">|</span>
        <span>Steps: <span className="text-zinc-300">{formatNumber(config.timesteps)}</span></span>
        <span className="text-zinc-600">|</span>
        <span>Max/ep: <span className="text-zinc-300">{formatNumber(config.max_steps)}</span></span>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <StatCard
          label="Total Episodes"
          value={formatNumber(analysis.episode_stats.total_episodes)}
          icon={GraduationCap}
        />
        <StatCard
          label="Mean Ep. Length"
          value={formatNumber(analysis.episode_stats.length.mean)}
          icon={Clock}
          subtitle={`median ${formatNumber(analysis.episode_stats.length.median)}`}
        />
        <StatCard
          label="Unique States"
          value={formatNumber(analysis.coverage.final_unique_states)}
          icon={Eye}
          subtitle={`${formatNumber(analysis.coverage.growth_rate_per_1k)}/1K rate`}
        />
        <StatCard
          label="Mean FPS"
          value={analysis.step_stats.fps.mean.toFixed(1)}
          icon={Gauge}
          subtitle={`min ${analysis.step_stats.fps.min.toFixed(1)}`}
        />
        <StatCard
          label="Paddle Status"
          value={analysis.paddle_analysis.status}
          icon={Activity}
          color={analysis.paddle_analysis.status === "HEALTHY" ? "text-emerald-400" : "text-red-400"}
        />
      </div>

      {/* Coverage chart */}
      <CoverageChart
        entries={analysis.coverage.entries}
        color={gameChartColor(config.game)}
      />

      {/* Episode stats breakdown */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Episode length distribution */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
          <h3 className="mb-4 text-sm font-medium text-zinc-300">Episode Length Stats</h3>
          <div className="space-y-2">
            {(["min", "median", "mean", "max", "std"] as const).map((key) => (
              <div key={key} className="flex items-center justify-between text-xs">
                <span className="text-zinc-500 capitalize">{key}</span>
                <span className="font-mono text-zinc-300">
                  {formatNumber(analysis.episode_stats.length[key], key === "std" ? 1 : 0)}
                </span>
              </div>
            ))}
            <div className="mt-2 border-t border-zinc-800 pt-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-zinc-500">Game over</span>
                <span className="font-mono text-zinc-300">
                  {analysis.episode_stats.game_over_count} ({((analysis.episode_stats.game_over_count / analysis.episode_stats.total_episodes) * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-zinc-500">Truncated</span>
                <span className="font-mono text-zinc-300">
                  {analysis.episode_stats.truncated_count} ({((analysis.episode_stats.truncated_count / analysis.episode_stats.total_episodes) * 100).toFixed(1)}%)
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Reward distribution */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
          <h3 className="mb-4 text-sm font-medium text-zinc-300">Reward Stats</h3>
          <div className="space-y-2">
            {(["min", "median", "mean", "max", "std"] as const).map((key) => (
              <div key={key} className="flex items-center justify-between text-xs">
                <span className="text-zinc-500 capitalize">{key}</span>
                <span className="font-mono text-zinc-300">
                  {analysis.episode_stats.reward[key].toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Degenerate episodes */}
      {analysis.degenerate_episodes.length > 0 && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-5">
          <h3 className="mb-3 text-sm font-medium text-amber-400">
            Degenerate Episodes ({analysis.degenerate_episodes.length})
          </h3>
          <div className="space-y-1.5">
            {analysis.degenerate_episodes.map((d) => (
              <div key={d.episode} className="flex items-center gap-4 text-xs">
                <span className="font-mono text-zinc-400">Ep {d.episode}</span>
                <span className="text-zinc-500">{d.steps} steps</span>
                <span className="text-zinc-500">{d.total_reward.toFixed(2)} reward</span>
                <span className="text-amber-400/80">{d.reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
