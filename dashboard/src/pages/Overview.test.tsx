import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { OverviewPage } from "@/pages/Overview";
import { MOCK_SESSIONS, MOCK_TRAINING_ANALYSES } from "@/data/mock";

function renderPage() {
  return render(
    <MemoryRouter>
      <OverviewPage />
    </MemoryRouter>,
  );
}

describe("OverviewPage", () => {
  it("renders the page heading", () => {
    renderPage();
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(
      screen.getByText("RL-driven autonomous game testing platform"),
    ).toBeInTheDocument();
  });

  it("shows the correct number of games tested", () => {
    renderPage();
    const uniqueGames = [...new Set(MOCK_SESSIONS.map((s) => s.game))];
    expect(screen.getByText(String(uniqueGames.length))).toBeInTheDocument();
    expect(screen.getByText("Games Tested")).toBeInTheDocument();
  });

  it("shows aggregated total findings across all sessions", () => {
    renderPage();
    const totalFindings = MOCK_SESSIONS.reduce(
      (sum, s) => sum + s.summary.total_findings,
      0,
    );
    expect(screen.getByText("Total Findings")).toBeInTheDocument();
    // formatNumber uses toLocaleString, look for the formatted value
    expect(screen.getByText(totalFindings.toLocaleString())).toBeInTheDocument();
  });

  it("shows aggregated critical findings count", () => {
    renderPage();
    const criticalFindings = MOCK_SESSIONS.reduce(
      (sum, s) => sum + s.summary.critical_findings,
      0,
    );
    expect(screen.getByText("Critical Bugs")).toBeInTheDocument();
    expect(screen.getByText(String(criticalFindings))).toBeInTheDocument();
  });

  it("shows total unique states from training analyses", () => {
    renderPage();
    const totalStates = MOCK_TRAINING_ANALYSES.reduce(
      (sum, a) => sum + a.coverage.final_unique_states,
      0,
    );
    expect(screen.getByText("Unique States")).toBeInTheDocument();
    expect(screen.getByText(totalStates.toLocaleString())).toBeInTheDocument();
  });

  it("renders the Evaluation Sessions table with all sessions", () => {
    renderPage();
    expect(screen.getByText("Evaluation Sessions")).toBeInTheDocument();
    // Each unique build_id should appear (some may appear multiple times across games)
    const uniqueBuilds = [...new Set(MOCK_SESSIONS.map((s) => s.build_id))];
    for (const buildId of uniqueBuilds) {
      const matches = screen.getAllByText(buildId);
      expect(matches.length).toBeGreaterThan(0);
    }
  });

  it("renders session table headers", () => {
    renderPage();
    expect(screen.getByText("Game")).toBeInTheDocument();
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("Episodes")).toBeInTheDocument();
    expect(screen.getByText("Mean Length")).toBeInTheDocument();
    expect(screen.getByText("Mean Reward")).toBeInTheDocument();
  });

  it("renders critical findings section when critical findings exist", () => {
    renderPage();
    const hasCritical = MOCK_SESSIONS.some((s) =>
      s.episodes.some((ep) =>
        ep.findings.some((f) => f.severity === "critical"),
      ),
    );
    if (hasCritical) {
      expect(screen.getByText("Critical Findings")).toBeInTheDocument();
    }
  });

  it("shows finding descriptions in the critical findings section", () => {
    renderPage();
    const criticalFindings = MOCK_SESSIONS.flatMap((s) =>
      s.episodes.flatMap((ep) =>
        ep.findings.filter((f) => f.severity === "critical"),
      ),
    );
    // Descriptions may appear multiple times (e.g. same freeze message across episodes)
    const uniqueDescriptions = [...new Set(criticalFindings.map((f) => f.description))];
    for (const desc of uniqueDescriptions) {
      const matches = screen.getAllByText(desc);
      expect(matches.length).toBeGreaterThan(0);
    }
  });

  it("renders GameBadge for each game in the sessions table", () => {
    renderPage();
    // GameBadges render in both the table and the critical findings section
    const breakoutBadges = screen.getAllByText("Breakout 71");
    expect(breakoutBadges.length).toBeGreaterThan(0);
    const hextrisBadges = screen.getAllByText("Hextris");
    expect(hextrisBadges.length).toBeGreaterThan(0);
  });

  it("renders mean episode length for each session", () => {
    renderPage();
    for (const session of MOCK_SESSIONS) {
      expect(
        screen.getByText(session.summary.mean_episode_length.toFixed(1)),
      ).toBeInTheDocument();
    }
  });
});
