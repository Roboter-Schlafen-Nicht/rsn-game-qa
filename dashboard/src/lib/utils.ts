import type { Severity } from "@/types";

/** Format a number with locale-aware separators */
export function formatNumber(n: number, decimals = 0): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Format a duration in seconds to a human-readable string */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

/** Severity color classes for Tailwind */
export function severityColor(severity: Severity): string {
  switch (severity) {
    case "critical":
      return "text-red-400";
    case "warning":
      return "text-amber-400";
    case "info":
    default:
      return "text-blue-400";
  }
}

/** Severity background color classes */
export function severityBgColor(severity: Severity): string {
  switch (severity) {
    case "critical":
      return "bg-red-500/10 text-red-400 border-red-500/20";
    case "warning":
      return "bg-amber-500/10 text-amber-400 border-amber-500/20";
    case "info":
    default:
      return "bg-blue-500/10 text-blue-400 border-blue-500/20";
  }
}

/** Game display name */
export function gameDisplayName(game: string): string {
  const names: Record<string, string> = {
    breakout71: "Breakout 71",
    hextris: "Hextris",
    shapez: "shapez.io",
  };
  return names[game] ?? game;
}

/** Game color class */
export function gameColor(game: string): string {
  const colors: Record<string, string> = {
    breakout71: "text-violet-400",
    hextris: "text-cyan-400",
    shapez: "text-emerald-400",
  };
  return colors[game] ?? "text-zinc-400";
}

/** Game background color for charts */
export function gameChartColor(game: string): string {
  const colors: Record<string, string> = {
    breakout71: "#8b5cf6",
    hextris: "#06b6d4",
    shapez: "#10b981",
  };
  return colors[game] ?? "#71717a";
}
