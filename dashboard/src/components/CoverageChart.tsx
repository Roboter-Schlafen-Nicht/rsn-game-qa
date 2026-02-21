import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import type { CoverageEntry } from "@/types";
import { formatNumber } from "@/lib/utils";

interface CoverageChartProps {
  entries: CoverageEntry[];
  color?: string;
  height?: number;
}

export function CoverageChart({ entries, color = "#8b5cf6", height = 240 }: CoverageChartProps) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
      <h3 className="mb-4 text-sm font-medium text-zinc-300">State Coverage Over Training</h3>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={entries} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="step"
            tickFormatter={(v: number) => `${v / 1000}K`}
            stroke="#52525b"
            fontSize={11}
          />
          <YAxis
            tickFormatter={(v: number) => formatNumber(v)}
            stroke="#52525b"
            fontSize={11}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: "8px",
              fontSize: 12,
            }}
            labelFormatter={(v) => `Step ${formatNumber(Number(v))}`}
            formatter={(v) => [formatNumber(Number(v)), "Unique States"]}
          />
          <Line
            type="monotone"
            dataKey="unique_states"
            stroke={color}
            strokeWidth={2}
            dot={{ r: 3, fill: color }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
