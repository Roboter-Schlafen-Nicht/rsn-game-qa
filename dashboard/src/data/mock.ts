import type { SessionReport } from "@/types/reporting";
import type { TrainingAnalysisReport } from "@/types/analysis";

/**
 * Mock session reports based on real evaluation data from Phases 1-4.
 * One entry per game x model type evaluated.
 */
export const MOCK_SESSIONS: SessionReport[] = [
  // Breakout 71 — trained model (Phase 2c, 200K steps)
  {
    session_id: "b71-trained-200k",
    game: "breakout71",
    build_id: "ppo_breakout71_200003",
    timestamp: "2026-02-19T14:30:00Z",
    episodes: Array.from({ length: 10 }, (_, i) => ({
      episode_id: i,
      steps: i < 9 ? [4, 3, 4, 5, 3, 4, 6, 3, 4][i] : 2000,
      total_reward: i < 9 ? -5.01 : 15.42,
      terminated: i < 9,
      truncated: i === 9,
      findings: i === 3
        ? [
            {
              oracle_name: "crash",
              severity: "critical" as const,
              step: 2,
              description: "Frozen frame detected: 16 consecutive identical frames",
              data: { frozen_frames: 16, duration_ms: 1066 },
              screenshot_path: null,
            },
          ]
        : i < 9
          ? []
          : [
              {
                oracle_name: "performance",
                severity: "warning" as const,
                step: 1500,
                description: "RAM usage exceeded 4GB threshold",
                data: { ram_mb: 4312 },
                screenshot_path: null,
              },
            ],
      metrics: {
        mean_fps: 14.97,
        min_fps: 12.1,
        max_reward_per_step: i < 9 ? 0.01 : 0.03,
        total_duration_seconds: i < 9 ? 0.3 : 133.5,
      },
      seed: null,
    })),
    summary: {
      total_episodes: 10,
      total_findings: 2055,
      critical_findings: 1,
      warning_findings: 270,
      info_findings: 1784,
      episodes_failed: 9,
      mean_episode_reward: -2.48,
      mean_episode_length: 203.8,
    },
  },
  // Breakout 71 — random baseline
  {
    session_id: "b71-random",
    game: "breakout71",
    build_id: "random_baseline",
    timestamp: "2026-02-19T15:00:00Z",
    episodes: Array.from({ length: 10 }, (_, i) => ({
      episode_id: i,
      steps: i < 7 ? [3, 15, 8, 5, 22, 4, 9][i] : 2000,
      total_reward: i < 7 ? -5.01 : 18.99,
      terminated: i < 7,
      truncated: i >= 7,
      findings: [],
      metrics: {
        mean_fps: 14.5,
        min_fps: 11.8,
        max_reward_per_step: 0.01,
        total_duration_seconds: i < 7 ? 0.5 : 133.5,
      },
      seed: null,
    })),
    summary: {
      total_episodes: 10,
      total_findings: 11804,
      critical_findings: 0,
      warning_findings: 0,
      info_findings: 11804,
      episodes_failed: 7,
      mean_episode_reward: 2.57,
      mean_episode_length: 608.4,
    },
  },
  // Hextris — trained model (Phase 4, 200K steps)
  {
    session_id: "hextris-trained-200k",
    game: "hextris",
    build_id: "ppo_hextris_200704",
    timestamp: "2026-02-19T18:30:00Z",
    episodes: Array.from({ length: 10 }, (_, i) => ({
      episode_id: i,
      steps: [380, 520, 290, 410, 350, 470, 380, 440, 310, 490][i],
      total_reward: [-1.2, 0.5, -2.1, -0.8, -1.5, 0.2, -1.0, -0.3, -1.8, 0.2][i],
      terminated: true,
      truncated: false,
      findings:
        i === 2 || i === 5 || i === 8
          ? [
              {
                oracle_name: "crash",
                severity: "critical" as const,
                step: [150, 300, 200][i === 2 ? 0 : i === 5 ? 1 : 2],
                description: "Frozen frame detected: 18 consecutive identical frames",
                data: { frozen_frames: 18, duration_ms: 1200 },
                screenshot_path: null,
              },
            ]
          : [],
      metrics: {
        mean_fps: 19.2,
        min_fps: 16.5,
        max_reward_per_step: 0.02,
        total_duration_seconds: [20, 27, 15, 21, 18, 25, 20, 23, 16, 26][i],
      },
      seed: null,
    })),
    summary: {
      total_episodes: 10,
      total_findings: 4741,
      critical_findings: 3,
      warning_findings: 735,
      info_findings: 4003,
      episodes_failed: 10,
      mean_episode_reward: -0.98,
      mean_episode_length: 404,
    },
  },
  // Hextris — random baseline
  {
    session_id: "hextris-random",
    game: "hextris",
    build_id: "random_baseline",
    timestamp: "2026-02-19T19:00:00Z",
    episodes: Array.from({ length: 10 }, (_, i) => ({
      episode_id: i,
      steps: [350, 480, 270, 390, 330, 440, 360, 410, 290, 420][i],
      total_reward: [-1.5, -0.8, -2.3, -1.1, -1.7, -0.6, -1.3, -0.9, -2.0, -0.6][i],
      terminated: true,
      truncated: false,
      findings: [],
      metrics: {
        mean_fps: 19.0,
        min_fps: 16.0,
        max_reward_per_step: 0.01,
        total_duration_seconds: [18, 25, 14, 21, 17, 23, 19, 22, 15, 22][i],
      },
      seed: null,
    })),
    summary: {
      total_episodes: 10,
      total_findings: 4186,
      critical_findings: 0,
      warning_findings: 462,
      info_findings: 3724,
      episodes_failed: 10,
      mean_episode_reward: -1.28,
      mean_episode_length: 374,
    },
  },
];

/**
 * Mock training analysis reports from real training runs.
 */
export const MOCK_TRAINING_ANALYSES: TrainingAnalysisReport[] = [
  // Breakout 71 — 200K CNN+RND
  {
    config: {
      game: "breakout71",
      policy: "cnn",
      reward_mode: "rnd",
      timesteps: 200000,
      max_steps: 2000,
    },
    episode_stats: {
      total_episodes: 259,
      game_over_count: 231,
      truncated_count: 28,
      length: { count: 259, mean: 305, median: 35, min: 6, max: 2000, std: 520 },
      reward: { count: 259, mean: 1.18, median: -4.5, min: -5.01, max: 15.42, std: 5.3 },
    },
    step_stats: {
      total_steps: 200003,
      fps: { count: 200003, mean: 14.97, median: 15.1, min: 8.2, max: 19.5, std: 1.8 },
    },
    coverage: {
      entries: [
        { step: 10000, unique_states: 1520 },
        { step: 20000, unique_states: 3180 },
        { step: 50000, unique_states: 6920 },
        { step: 100000, unique_states: 10250 },
        { step: 150000, unique_states: 13100 },
        { step: 200000, unique_states: 15521 },
      ],
      final_unique_states: 15521,
      growth_rate_per_1k: 214,
    },
    paddle_analysis: {
      paddle_x: {
        count: 200003,
        mean: 0.52,
        median: 0.48,
        min: 0.05,
        max: 0.95,
        std: 0.21,
        unique_positions: 9,
        most_common_pct: 55.3,
      },
      status: "HEALTHY",
    },
    per_episode_rnd: [
      { episode: 0, steps: 1955, rnd_mean: 0.0027, rnd_total: 5.28, termination: "game_over", rnd_collapsed: true },
      { episode: 1, steps: 629, rnd_mean: 0.0009, rnd_total: 0.57, termination: "game_over", rnd_collapsed: true },
      { episode: 2, steps: 49, rnd_mean: 0.0031, rnd_total: 0.15, termination: "game_over", rnd_collapsed: false },
    ],
    degenerate_episodes: [
      { episode: 12, steps: 2000, total_reward: 3.7, termination: "truncated", reason: "max_steps truncation" },
      { episode: 45, steps: 2000, total_reward: 3.8, termination: "truncated", reason: "max_steps truncation" },
    ],
  },
  // Hextris — 200K CNN
  {
    config: {
      game: "hextris",
      policy: "cnn",
      reward_mode: "survival",
      timesteps: 200000,
      max_steps: 2000,
    },
    episode_stats: {
      total_episodes: 323,
      game_over_count: 323,
      truncated_count: 0,
      length: { count: 323, mean: 620, median: 580, min: 45, max: 1850, std: 380 },
      reward: { count: 323, mean: 1.18, median: 0.8, min: -2.45, max: 9.1, std: 1.78 },
    },
    step_stats: {
      total_steps: 200704,
      fps: { count: 200704, mean: 19.0, median: 19.2, min: 14.5, max: 22.1, std: 1.2 },
    },
    coverage: {
      entries: [
        { step: 10000, unique_states: 12500 },
        { step: 20000, unique_states: 28000 },
        { step: 50000, unique_states: 62000 },
        { step: 100000, unique_states: 108000 },
        { step: 150000, unique_states: 148000 },
        { step: 200000, unique_states: 184314 },
      ],
      final_unique_states: 184314,
      growth_rate_per_1k: 920,
    },
    paddle_analysis: {
      paddle_x: {
        count: 200704,
        mean: 0.5,
        median: 0.5,
        min: 0.0,
        max: 1.0,
        std: 0.33,
        unique_positions: 3,
        most_common_pct: 34.2,
      },
      status: "HEALTHY",
    },
    per_episode_rnd: [],
    degenerate_episodes: [],
  },
];

/**
 * Cross-game comparison data for the comparison page.
 */
export interface CrossGameComparison {
  game: string;
  trained: {
    steps: number;
    episodes: number;
    uniqueStates: number;
    meanLength: number;
    meanReward: number;
    criticalFindings: number;
    totalFindings: number;
  };
  random: {
    meanLength: number;
    meanReward: number;
    criticalFindings: number;
    totalFindings: number;
  };
  vsRandom: {
    lengthRatio: number;
    criticalDelta: number;
  };
}

export const MOCK_CROSS_GAME: CrossGameComparison[] = [
  {
    game: "breakout71",
    trained: {
      steps: 200000,
      episodes: 259,
      uniqueStates: 15521,
      meanLength: 203.8,
      meanReward: -2.48,
      criticalFindings: 1,
      totalFindings: 2055,
    },
    random: {
      meanLength: 608.4,
      meanReward: 2.57,
      criticalFindings: 0,
      totalFindings: 11804,
    },
    vsRandom: {
      lengthRatio: 0.34,
      criticalDelta: 1,
    },
  },
  {
    game: "hextris",
    trained: {
      steps: 200000,
      episodes: 323,
      uniqueStates: 184314,
      meanLength: 404,
      meanReward: -0.98,
      criticalFindings: 3,
      totalFindings: 4741,
    },
    random: {
      meanLength: 374,
      meanReward: -1.28,
      criticalFindings: 0,
      totalFindings: 4186,
    },
    vsRandom: {
      lengthRatio: 1.08,
      criticalDelta: 3,
    },
  },
];
