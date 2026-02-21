/**
 * Training log event types — parsed from JSONL output of train_rl.py.
 * Each line is a JSON object with an "event" discriminator field.
 */

/** Initial config event logged at training start */
export interface ConfigEvent {
  event: "config";
  timestamp: string;
  game: string;
  args: {
    timesteps: number;
    policy: "cnn" | "mlp";
    reward_mode: "yolo" | "survival" | "rnd";
    max_steps: number;
    frame_stack: number;
    n_steps: number;
    batch_size: number;
    n_epochs: number;
    gamma: number;
    lr: number;
    clip_range: number;
    ent_coef: number;
    headless: boolean;
    orientation: "portrait" | "landscape";
    survival_bonus: number;
    epsilon_greedy: number;
    rnd_int_coeff: number;
    rnd_ext_coeff: number;
    [key: string]: unknown;
  };
  browser: string;
  python_version: string;
  platform: string;
}

/** Per-step summary event */
export interface StepSummaryEvent {
  event: "step_summary";
  timestamp: string;
  step: number;
  episode: number;
  reward: number;
  cumulative_reward: number;
  ball_detected: boolean;
  brick_count: number;
  paddle_x: number;
  action: number | number[];
  fps: number;
  no_ball_count: number;
  rnd_intrinsic_raw?: number;
  rnd_intrinsic_norm?: number;
}

/** Episode completion event */
export interface EpisodeEndEvent {
  event: "episode_end";
  timestamp: string;
  episode: number;
  steps: number;
  total_reward: number;
  mean_step_reward: number;
  termination: "game_over" | "truncated";
  brick_count: number;
  duration_seconds: number;
  mean_fps: number;
  rnd_intrinsic_total?: number;
  rnd_intrinsic_mean?: number;
}

/** Periodic coverage summary */
export interface CoverageSummaryEvent {
  event: "coverage_summary";
  timestamp: string;
  step: number;
  unique_states: number;
}

/** Max time reached shutdown event */
export interface MaxTimeReachedEvent {
  event: "max_time_reached";
  timestamp: string;
  step: number;
  elapsed_seconds: number;
  max_time_seconds: number;
}

/** Discriminated union of all training log event types */
export type TrainingLogEvent =
  | ConfigEvent
  | StepSummaryEvent
  | EpisodeEndEvent
  | CoverageSummaryEvent
  | MaxTimeReachedEvent;
