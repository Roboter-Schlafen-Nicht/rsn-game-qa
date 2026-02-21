/**
 * Training analysis report — output of analyze_training.py --json.
 * Matches the Python build_analysis() output format.
 */

/** Descriptive statistics for a numeric field */
export interface DescribeStats {
  count: number;
  mean: number;
  median: number;
  min: number;
  max: number;
  std: number;
}

/** Paddle-specific descriptive stats with unique position count */
export interface PaddleDescribeStats extends DescribeStats {
  unique_positions: number;
  most_common_pct: number;
}

/** Coverage measurement at a point in training */
export interface CoverageEntry {
  step: number;
  unique_states: number;
}

/** Per-episode RND analysis */
export interface EpisodeRndAnalysis {
  episode: number;
  steps: number;
  rnd_mean: number;
  rnd_total: number;
  termination: "game_over" | "truncated";
  rnd_collapsed: boolean;
}

/** Degenerate episode detection */
export interface DegenerateEpisode {
  episode: number;
  steps: number;
  total_reward: number;
  termination: "game_over" | "truncated";
  reason: string;
}

/** Training config section */
export interface AnalysisConfig {
  game: string;
  policy: string;
  reward_mode: string;
  timesteps: number;
  max_steps: number;
  [key: string]: unknown;
}

/** Episode-level statistics */
export interface EpisodeStats {
  total_episodes: number;
  game_over_count: number;
  truncated_count: number;
  length: DescribeStats;
  reward: DescribeStats;
}

/** Step-level statistics */
export interface StepStats {
  total_steps: number;
  fps: DescribeStats;
}

/** Coverage analysis section */
export interface CoverageAnalysis {
  entries: CoverageEntry[];
  final_unique_states: number;
  growth_rate_per_1k: number;
}

/** Paddle movement analysis */
export interface PaddleAnalysis {
  paddle_x: PaddleDescribeStats;
  status: "HEALTHY" | "DEGENERATE" | "UNKNOWN";
}

/** Top-level training analysis report */
export interface TrainingAnalysisReport {
  config: AnalysisConfig;
  episode_stats: EpisodeStats;
  step_stats: StepStats;
  coverage: CoverageAnalysis;
  paddle_analysis: PaddleAnalysis;
  per_episode_rnd: EpisodeRndAnalysis[];
  degenerate_episodes: DegenerateEpisode[];
}
