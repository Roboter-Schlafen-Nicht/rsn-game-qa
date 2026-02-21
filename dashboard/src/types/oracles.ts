/**
 * Oracle type definitions — matches the 12 oracle implementations
 * in src/oracles/.
 */

/** All oracle names as a literal union */
export type OracleName =
  | "crash"
  | "stuck"
  | "visual_glitch"
  | "performance"
  | "score_anomaly"
  | "physics_violation"
  | "boundary"
  | "state_transition"
  | "episode_length"
  | "temporal_anomaly"
  | "reward_consistency"
  | "soak";

/** Oracle display metadata */
export interface OracleInfo {
  name: OracleName;
  label: string;
  description: string;
  severity: "critical" | "warning" | "info";
}

/** All oracle metadata for display */
export const ORACLE_INFO: Record<OracleName, OracleInfo> = {
  crash: {
    name: "crash",
    label: "Crash",
    description: "Detects frozen frames and application crashes",
    severity: "critical",
  },
  stuck: {
    name: "stuck",
    label: "Stuck State",
    description: "Detects when the game enters an unresponsive state",
    severity: "critical",
  },
  visual_glitch: {
    name: "visual_glitch",
    label: "Visual Glitch",
    description: "Detects rendering anomalies and visual artifacts",
    severity: "warning",
  },
  performance: {
    name: "performance",
    label: "Performance",
    description: "Monitors FPS drops and resource usage",
    severity: "warning",
  },
  score_anomaly: {
    name: "score_anomaly",
    label: "Score Anomaly",
    description: "Detects unexpected score changes",
    severity: "warning",
  },
  physics_violation: {
    name: "physics_violation",
    label: "Physics Violation",
    description: "Detects physics engine anomalies",
    severity: "critical",
  },
  boundary: {
    name: "boundary",
    label: "Boundary",
    description: "Detects objects leaving valid play area",
    severity: "warning",
  },
  state_transition: {
    name: "state_transition",
    label: "State Transition",
    description: "Detects invalid game state transitions",
    severity: "warning",
  },
  episode_length: {
    name: "episode_length",
    label: "Episode Length",
    description: "Flags abnormally short or long episodes",
    severity: "info",
  },
  temporal_anomaly: {
    name: "temporal_anomaly",
    label: "Temporal Anomaly",
    description: "Detects timing and frame rate irregularities",
    severity: "info",
  },
  reward_consistency: {
    name: "reward_consistency",
    label: "Reward Consistency",
    description: "Detects reward signal anomalies",
    severity: "info",
  },
  soak: {
    name: "soak",
    label: "Soak Test",
    description: "Long-running stability and memory leak detection",
    severity: "info",
  },
};
