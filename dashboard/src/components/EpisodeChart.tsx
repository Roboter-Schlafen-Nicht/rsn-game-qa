import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts";
import type { EpisodeReport } from "@/types";

interface EpisodeChartProps {
  episodes: EpisodeReport[];
  height?: number;
}

export function EpisodeChart({ episodes, height = 240 }: EpisodeChartProps) {
  const data = episodes.map((ep) => ({
    id: `E${ep.episode_id}`,
    steps: ep.steps,
    reward: ep.total_reward,
    terminated: ep.terminated,
    truncated: ep.truncated,
  }));

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
      <h3 className="mb-4 text-sm font-medium text-zinc-300">Episode Length</h3>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
          <XAxis dataKey="id" stroke="#52525b" fontSize={11} />
          <YAxis stroke="#52525b" fontSize={11} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: "8px",
              fontSize: 12,
            }}
            formatter={(v, name) => [v, name === "steps" ? "Steps" : name]}
          />
          <Bar dataKey="steps" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.truncated ? "#f59e0b" : entry.steps < 10 ? "#ef4444" : "#8b5cf6"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-3 flex gap-4 text-[11px] text-zinc-500">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-violet-500" /> Normal
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-red-500" /> Instant death
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-amber-500" /> Truncated
        </span>
      </div>
    </div>
  );
}
