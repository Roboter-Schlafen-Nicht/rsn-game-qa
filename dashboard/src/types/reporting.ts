/** Severity levels matching Python FindingReport.severity */
export type Severity = "critical" | "warning" | "info";

/** Single finding from an oracle during evaluation */
export interface FindingReport {
  oracle_name: string;
  severity: Severity;
  step: number;
  description: string;
  data: Record<string, unknown>;
  screenshot_path: string | null;
}

/** Per-episode performance metrics */
export interface EpisodeMetrics {
  mean_fps: number | null;
  min_fps: number | null;
  max_reward_per_step: number | null;
  total_duration_seconds: number | null;
}

/** Single episode within an evaluation session */
export interface EpisodeReport {
  episode_id: number;
  steps: number;
  total_reward: number;
  terminated: boolean;
  truncated: boolean;
  findings: FindingReport[];
  metrics: EpisodeMetrics;
  seed: number | null;
}

/** Aggregate stats for the entire session */
export interface SessionSummary {
  total_episodes: number;
  total_findings: number;
  critical_findings: number;
  warning_findings: number;
  info_findings: number;
  episodes_failed: number;
  mean_episode_reward: number;
  mean_episode_length: number;
}

/** Top-level session report — output of run_session.py */
export interface SessionReport {
  session_id: string;
  game: string;
  build_id: string;
  timestamp: string;
  episodes: EpisodeReport[];
  summary: SessionSummary;
}
