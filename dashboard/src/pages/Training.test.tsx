import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { TrainingPage } from "@/pages/Training";
import { MOCK_TRAINING_ANALYSES } from "@/data/mock";
import { gameDisplayName, formatNumber } from "@/lib/utils";

// Mock Recharts — ResponsiveContainer needs real DOM dimensions
vi.mock("recharts", async () => {
  const actual = await vi.importActual<typeof import("recharts")>("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
  };
});

function renderPage() {
  return render(
    <MemoryRouter>
      <TrainingPage />
    </MemoryRouter>,
  );
}

describe("TrainingPage", () => {
  const defaultAnalysis = MOCK_TRAINING_ANALYSES[0];

  it("renders the page heading", () => {
    renderPage();
    expect(screen.getByText("Training")).toBeInTheDocument();
    expect(
      screen.getByText("Training run analysis and metrics"),
    ).toBeInTheDocument();
  });

  it("renders game selector buttons for each training analysis", () => {
    renderPage();
    for (const analysis of MOCK_TRAINING_ANALYSES) {
      expect(
        screen.getByRole("button", {
          name: gameDisplayName(analysis.config.game),
        }),
      ).toBeInTheDocument();
    }
  });

  it("shows config summary for the default (first) game", () => {
    renderPage();
    expect(
      screen.getByText(defaultAnalysis.config.policy.toUpperCase()),
    ).toBeInTheDocument();
    expect(
      screen.getByText(defaultAnalysis.config.reward_mode),
    ).toBeInTheDocument();
    expect(
      screen.getByText(formatNumber(defaultAnalysis.config.timesteps)),
    ).toBeInTheDocument();
  });

  it("shows total episodes stat card", () => {
    renderPage();
    expect(screen.getByText("Total Episodes")).toBeInTheDocument();
    expect(
      screen.getByText(
        formatNumber(defaultAnalysis.episode_stats.total_episodes),
      ),
    ).toBeInTheDocument();
  });

  it("shows unique states stat card with growth rate", () => {
    renderPage();
    expect(screen.getByText("Unique States")).toBeInTheDocument();
    expect(
      screen.getByText(formatNumber(defaultAnalysis.coverage.final_unique_states)),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        `${formatNumber(defaultAnalysis.coverage.growth_rate_per_1k)}/1K rate`,
      ),
    ).toBeInTheDocument();
  });

  it("shows paddle analysis status", () => {
    renderPage();
    expect(screen.getByText("Paddle Status")).toBeInTheDocument();
    expect(
      screen.getByText(defaultAnalysis.paddle_analysis.status),
    ).toBeInTheDocument();
  });

  it("shows episode length stats breakdown", () => {
    renderPage();
    expect(screen.getByText("Episode Length Stats")).toBeInTheDocument();
    // Check game_over/truncated counts
    const goCount = defaultAnalysis.episode_stats.game_over_count;
    const goPercent = (
      (goCount / defaultAnalysis.episode_stats.total_episodes) *
      100
    ).toFixed(1);
    expect(
      screen.getByText(`${goCount} (${goPercent}%)`),
    ).toBeInTheDocument();
  });

  it("shows reward stats breakdown", () => {
    renderPage();
    expect(screen.getByText("Reward Stats")).toBeInTheDocument();
  });

  it("shows degenerate episodes section when present", () => {
    renderPage();
    if (defaultAnalysis.degenerate_episodes.length > 0) {
      expect(
        screen.getByText(
          `Degenerate Episodes (${defaultAnalysis.degenerate_episodes.length})`,
        ),
      ).toBeInTheDocument();
      const reasons: string[] = defaultAnalysis.degenerate_episodes.map((d: { reason: string }) => d.reason);
      const uniqueReasons = [...new Set(reasons)];
      for (const reason of uniqueReasons) {
        const count = reasons.filter((r) => r === reason).length;
        expect(screen.getAllByText(reason)).toHaveLength(count);
      }
    }
  });

  it("switches game when clicking a different game button", async () => {
    if (MOCK_TRAINING_ANALYSES.length < 2) return;

    const user = userEvent.setup();
    renderPage();

    const secondAnalysis = MOCK_TRAINING_ANALYSES[1];
    const secondGameButton = screen.getByRole("button", {
      name: gameDisplayName(secondAnalysis.config.game),
    });

    await user.click(secondGameButton);

    // After clicking, the second game's config should be visible
    expect(
      screen.getByText(secondAnalysis.config.policy.toUpperCase()),
    ).toBeInTheDocument();
    expect(
      screen.getByText(secondAnalysis.config.reward_mode),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        formatNumber(secondAnalysis.episode_stats.total_episodes),
      ),
    ).toBeInTheDocument();
  });

  it("renders the coverage chart container", () => {
    renderPage();
    expect(
      screen.getByText("State Coverage Over Training"),
    ).toBeInTheDocument();
  });
});
