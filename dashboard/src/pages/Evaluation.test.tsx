import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { EvaluationPage } from "@/pages/Evaluation";
import { MOCK_SESSIONS } from "@/data/mock";
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
      <EvaluationPage />
    </MemoryRouter>,
  );
}

describe("EvaluationPage", () => {
  const defaultSession = MOCK_SESSIONS[0];

  it("renders the page heading", () => {
    renderPage();
    expect(screen.getByText("Evaluation")).toBeInTheDocument();
    expect(
      screen.getByText("Session evaluation results and episode analysis"),
    ).toBeInTheDocument();
  });

  it("renders session selector buttons for all sessions", () => {
    renderPage();
    for (const s of MOCK_SESSIONS) {
      const label = `${gameDisplayName(s.game)} - ${s.build_id.includes("random") ? "Random" : "Trained"}`;
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
  });

  it("shows session info bar with game, model, and session ID", () => {
    renderPage();
    expect(screen.getByText(defaultSession.build_id)).toBeInTheDocument();
    expect(screen.getByText(defaultSession.session_id)).toBeInTheDocument();
  });

  it("shows stat cards for the default session", () => {
    renderPage();
    // "Episodes" appears as both a stat card label and table heading
    expect(screen.getAllByText("Episodes").length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getByText(String(defaultSession.summary.total_episodes)),
    ).toBeInTheDocument();
    expect(screen.getByText("Mean Length")).toBeInTheDocument();
    expect(
      screen.getByText(defaultSession.summary.mean_episode_length.toFixed(1)),
    ).toBeInTheDocument();
    expect(screen.getByText("Mean Reward")).toBeInTheDocument();
    expect(
      screen.getByText(defaultSession.summary.mean_episode_reward.toFixed(2)),
    ).toBeInTheDocument();
    expect(screen.getByText("Total Findings")).toBeInTheDocument();
    expect(
      screen.getByText(formatNumber(defaultSession.summary.total_findings)),
    ).toBeInTheDocument();
  });

  it("renders the episodes table with correct headers", () => {
    renderPage();
    expect(screen.getByText("Episode")).toBeInTheDocument();
    expect(screen.getByText("Steps")).toBeInTheDocument();
    expect(screen.getByText("Reward")).toBeInTheDocument();
    expect(screen.getByText("Status")).toBeInTheDocument();
    expect(screen.getByText("FPS")).toBeInTheDocument();
    expect(screen.getByText("Duration")).toBeInTheDocument();
  });

  it("renders episode rows for the default session", () => {
    renderPage();
    for (const ep of defaultSession.episodes) {
      expect(screen.getByText(`E${ep.episode_id}`)).toBeInTheDocument();
    }
  });

  it("shows terminated/truncated status labels", () => {
    renderPage();
    const hasTerminated = defaultSession.episodes.some(
      (ep) => ep.terminated && !ep.truncated,
    );
    const hasTruncated = defaultSession.episodes.some((ep) => ep.truncated);
    if (hasTerminated) {
      expect(screen.getAllByText("terminated").length).toBeGreaterThan(0);
    }
    if (hasTruncated) {
      expect(screen.getAllByText("truncated").length).toBeGreaterThan(0);
    }
  });

  it("renders session findings section when findings exist", () => {
    renderPage();
    const allFindings = defaultSession.episodes.flatMap((ep) => ep.findings);
    if (allFindings.length > 0) {
      expect(screen.getByText("Session Findings")).toBeInTheDocument();
    }
  });

  it("renders episode chart container", () => {
    renderPage();
    expect(screen.getByText("Episode Length")).toBeInTheDocument();
  });

  it("switches session when clicking a different session button", async () => {
    if (MOCK_SESSIONS.length < 2) return;

    const user = userEvent.setup();
    renderPage();

    const secondSession = MOCK_SESSIONS[1];
    const label = `${gameDisplayName(secondSession.game)} - ${secondSession.build_id.includes("random") ? "Random" : "Trained"}`;

    await user.click(screen.getByRole("button", { name: label }));

    // After switching, the second session's info should be visible
    expect(screen.getByText(secondSession.build_id)).toBeInTheDocument();
    expect(screen.getByText(secondSession.session_id)).toBeInTheDocument();
    expect(
      screen.getByText(secondSession.summary.mean_episode_length.toFixed(1)),
    ).toBeInTheDocument();
  });
});
