import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { CrossGamePage } from "@/pages/CrossGame";
import { MOCK_CROSS_GAME } from "@/data/mock";
import { formatNumber, gameDisplayName } from "@/lib/utils";

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
      <CrossGamePage />
    </MemoryRouter>,
  );
}

describe("CrossGamePage", () => {
  it("renders the page heading", () => {
    renderPage();
    expect(
      screen.getByText("Cross-Game Comparison"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Side-by-side analysis across all tested games"),
    ).toBeInTheDocument();
  });

  it("shows games tested stat card", () => {
    renderPage();
    expect(screen.getByText("Games Tested")).toBeInTheDocument();
    expect(
      screen.getByText(String(MOCK_CROSS_GAME.length)),
    ).toBeInTheDocument();
  });

  it("shows total unique states across all games", () => {
    renderPage();
    const totalStates = MOCK_CROSS_GAME.reduce(
      (s, g) => s + g.trained.uniqueStates,
      0,
    );
    expect(screen.getByText("Total Unique States")).toBeInTheDocument();
    expect(
      screen.getByText(formatNumber(totalStates)),
    ).toBeInTheDocument();
  });

  it("shows critical bugs found stat card", () => {
    renderPage();
    const totalCritical = MOCK_CROSS_GAME.reduce(
      (s, g) => s + g.trained.criticalFindings,
      0,
    );
    expect(screen.getByText("Critical Bugs Found")).toBeInTheDocument();
    expect(
      screen.getByText(String(totalCritical)),
    ).toBeInTheDocument();
  });

  it("shows total training steps stat card", () => {
    renderPage();
    const totalSteps = MOCK_CROSS_GAME.reduce(
      (s, g) => s + g.trained.steps,
      0,
    );
    expect(screen.getByText("Total Training Steps")).toBeInTheDocument();
    expect(
      screen.getByText(formatNumber(totalSteps)),
    ).toBeInTheDocument();
  });

  it("renders game comparison table", () => {
    renderPage();
    expect(screen.getByText("Game Comparison")).toBeInTheDocument();
    // Table should have metric labels
    expect(screen.getByText("Training steps")).toBeInTheDocument();
    expect(screen.getByText("Training episodes")).toBeInTheDocument();
    expect(screen.getByText("Unique visual states")).toBeInTheDocument();
    expect(screen.getByText("Mean episode length")).toBeInTheDocument();
    expect(screen.getByText("Mean reward")).toBeInTheDocument();
    expect(screen.getByText("Critical findings")).toBeInTheDocument();
    expect(screen.getByText("vs Random (length)")).toBeInTheDocument();
  });

  it("shows trained/random sub-headers in comparison table", () => {
    renderPage();
    // Each game gets a "Trained" and "Random" sub-header
    const trainedHeaders = screen.getAllByText("Trained");
    const randomHeaders = screen.getAllByText("Random");
    expect(trainedHeaders.length).toBe(MOCK_CROSS_GAME.length);
    expect(randomHeaders.length).toBe(MOCK_CROSS_GAME.length);
  });

  it("shows game badges in the comparison table", () => {
    renderPage();
    for (const g of MOCK_CROSS_GAME) {
      const badges = screen.getAllByText(gameDisplayName(g.game));
      // At least one badge in the table + possibly one in stat cards
      expect(badges.length).toBeGreaterThan(0);
    }
  });

  it("shows vs random ratio with x suffix", () => {
    renderPage();
    for (const g of MOCK_CROSS_GAME) {
      expect(
        screen.getByText(`${g.vsRandom.lengthRatio.toFixed(2)}x`),
      ).toBeInTheDocument();
    }
  });

  it("renders mean episode length chart container", () => {
    renderPage();
    expect(screen.getByText("Mean Episode Length")).toBeInTheDocument();
  });

  it("renders unique visual states chart container", () => {
    renderPage();
    expect(
      screen.getByText("Unique Visual States (Training)"),
    ).toBeInTheDocument();
  });

  it("shows baseline labels in the vs Random row", () => {
    renderPage();
    const baselineLabels = screen.getAllByText("baseline");
    expect(baselineLabels.length).toBe(MOCK_CROSS_GAME.length);
  });
});
