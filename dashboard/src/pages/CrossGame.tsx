import React from "react";
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
import { GitCompare, Eye, AlertTriangle, Clock } from "lucide-react";
import { StatCard } from "@/components/StatCard";
import { GameBadge } from "@/components/GameBadge";
import { MOCK_CROSS_GAME } from "@/data/mock";
import { formatNumber, gameDisplayName, gameChartColor } from "@/lib/utils";

export function CrossGamePage() {
  // Prepare chart data
  const comparisonData = MOCK_CROSS_GAME.map((g) => ({
    game: gameDisplayName(g.game),
    "Trained Length": g.trained.meanLength,
    "Random Length": g.random.meanLength,
    "Trained Critical": g.trained.criticalFindings,
    "Random Critical": g.random.criticalFindings,
  }));

  const coverageData = MOCK_CROSS_GAME.map((g) => ({
    game: gameDisplayName(g.game),
    states: g.trained.uniqueStates,
    fill: gameChartColor(g.game),
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Cross-Game Comparison</h1>
        <p className="text-sm text-zinc-500">
          Side-by-side analysis across all tested games
        </p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          label="Games Tested"
          value={MOCK_CROSS_GAME.length}
          icon={GitCompare}
        />
        <StatCard
          label="Total Unique States"
          value={formatNumber(
            MOCK_CROSS_GAME.reduce((s, g) => s + g.trained.uniqueStates, 0),
          )}
          icon={Eye}
        />
        <StatCard
          label="Critical Bugs Found"
          value={MOCK_CROSS_GAME.reduce((s, g) => s + g.trained.criticalFindings, 0)}
          icon={AlertTriangle}
          color="text-red-400"
          subtitle="That random missed"
        />
        <StatCard
          label="Total Training Steps"
          value={formatNumber(
            MOCK_CROSS_GAME.reduce((s, g) => s + g.trained.steps, 0),
          )}
          icon={Clock}
        />
      </div>

      {/* Comparison table */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
        <div className="border-b border-zinc-800 px-5 py-3">
          <h2 className="text-sm font-medium text-zinc-300">Game Comparison</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left text-xs font-medium text-zinc-500">
                <th className="px-5 py-3">Metric</th>
                {MOCK_CROSS_GAME.map((g) => (
                  <th key={g.game} className="px-5 py-3" colSpan={2}>
                    <GameBadge game={g.game} />
                  </th>
                ))}
              </tr>
              <tr className="border-b border-zinc-800 text-left text-[10px] font-medium text-zinc-600">
                <th className="px-5 py-1" />
                {MOCK_CROSS_GAME.map((g) => (
                  <React.Fragment key={g.game}>
                    <th className="px-5 py-1">Trained</th>
                    <th className="px-5 py-1">Random</th>
                  </React.Fragment>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/50">
              <tr className="hover:bg-zinc-800/30">
                <td className="px-5 py-2.5 text-xs text-zinc-400">Training steps</td>
                {MOCK_CROSS_GAME.map((g) => (
                  <React.Fragment key={g.game}>
                    <td className="px-5 py-2.5 font-mono text-xs text-zinc-300">
                      {formatNumber(g.trained.steps)}
                    </td>
                    <td className="px-5 py-2.5 text-xs text-zinc-600">-</td>
                  </React.Fragment>
                ))}
              </tr>
              <tr className="hover:bg-zinc-800/30">
                <td className="px-5 py-2.5 text-xs text-zinc-400">Training episodes</td>
                {MOCK_CROSS_GAME.map((g) => (
                  <React.Fragment key={g.game}>
                    <td className="px-5 py-2.5 font-mono text-xs text-zinc-300">
                      {formatNumber(g.trained.episodes)}
                    </td>
                    <td className="px-5 py-2.5 text-xs text-zinc-600">-</td>
                  </React.Fragment>
                ))}
              </tr>
              <tr className="hover:bg-zinc-800/30">
                <td className="px-5 py-2.5 text-xs text-zinc-400">Unique visual states</td>
                {MOCK_CROSS_GAME.map((g) => (
                  <React.Fragment key={g.game}>
                    <td className="px-5 py-2.5 font-mono text-xs text-zinc-300">
                      {formatNumber(g.trained.uniqueStates)}
                    </td>
                    <td className="px-5 py-2.5 text-xs text-zinc-600">-</td>
                  </React.Fragment>
                ))}
              </tr>
              <tr className="hover:bg-zinc-800/30">
                <td className="px-5 py-2.5 text-xs text-zinc-400">Mean episode length</td>
                {MOCK_CROSS_GAME.map((g) => (
                  <React.Fragment key={g.game}>
                    <td className="px-5 py-2.5 font-mono text-xs text-zinc-300">
                      {g.trained.meanLength.toFixed(1)}
                    </td>
                    <td className="px-5 py-2.5 font-mono text-xs text-zinc-400">
                      {g.random.meanLength.toFixed(1)}
                    </td>
                  </React.Fragment>
                ))}
              </tr>
              <tr className="hover:bg-zinc-800/30">
                <td className="px-5 py-2.5 text-xs text-zinc-400">Mean reward</td>
                {MOCK_CROSS_GAME.map((g) => (
                  <React.Fragment key={g.game}>
                    <td className="px-5 py-2.5 font-mono text-xs text-zinc-300">
                      {g.trained.meanReward.toFixed(2)}
                    </td>
                    <td className="px-5 py-2.5 font-mono text-xs text-zinc-400">
                      {g.random.meanReward.toFixed(2)}
                    </td>
                  </React.Fragment>
                ))}
              </tr>
              <tr className="hover:bg-zinc-800/30">
                <td className="px-5 py-2.5 text-xs text-zinc-400">Critical findings</td>
                {MOCK_CROSS_GAME.map((g) => (
                  <React.Fragment key={g.game}>
                    <td className={`px-5 py-2.5 font-mono text-xs ${g.trained.criticalFindings > 0 ? "text-red-400" : "text-zinc-400"}`}>
                      {g.trained.criticalFindings}
                    </td>
                    <td className="px-5 py-2.5 font-mono text-xs text-zinc-400">
                      {g.random.criticalFindings}
                    </td>
                  </React.Fragment>
                ))}
              </tr>
              <tr className="hover:bg-zinc-800/30">
                <td className="px-5 py-2.5 text-xs text-zinc-400">vs Random (length)</td>
                {MOCK_CROSS_GAME.map((g) => (
                  <React.Fragment key={g.game}>
                    <td className={`px-5 py-2.5 font-mono text-xs font-medium ${g.vsRandom.lengthRatio >= 1 ? "text-emerald-400" : "text-red-400"}`}>
                      {g.vsRandom.lengthRatio.toFixed(2)}x
                    </td>
                    <td className="px-5 py-2.5 text-xs text-zinc-600">baseline</td>
                  </React.Fragment>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Episode length comparison */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
          <h3 className="mb-4 text-sm font-medium text-zinc-300">Mean Episode Length</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={comparisonData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
              <XAxis dataKey="game" stroke="#52525b" fontSize={11} />
              <YAxis stroke="#52525b" fontSize={11} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#18181b",
                  border: "1px solid #3f3f46",
                  borderRadius: "8px",
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="Trained Length" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Random Length" fill="#3f3f46" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* State coverage comparison */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
          <h3 className="mb-4 text-sm font-medium text-zinc-300">Unique Visual States (Training)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={coverageData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
              <XAxis dataKey="game" stroke="#52525b" fontSize={11} />
              <YAxis
                stroke="#52525b"
                fontSize={11}
                tickFormatter={(v) => `${(Number(v) / 1000).toFixed(0)}K`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#18181b",
                  border: "1px solid #3f3f46",
                  borderRadius: "8px",
                  fontSize: 12,
                }}
                formatter={(v) => [formatNumber(Number(v)), "States"]}
              />
              <Bar dataKey="states" radius={[4, 4, 0, 0]}>
                {coverageData.map((entry, index) => (
                  <Cell key={index} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
