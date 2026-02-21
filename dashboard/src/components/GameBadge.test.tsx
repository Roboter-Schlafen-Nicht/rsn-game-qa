import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { GameBadge } from "@/components/GameBadge";

describe("GameBadge", () => {
  it("renders display name for breakout71", () => {
    render(<GameBadge game="breakout71" />);
    expect(screen.getByText("Breakout 71")).toBeInTheDocument();
  });

  it("renders display name for hextris", () => {
    render(<GameBadge game="hextris" />);
    expect(screen.getByText("Hextris")).toBeInTheDocument();
  });

  it("renders display name for shapez", () => {
    render(<GameBadge game="shapez" />);
    expect(screen.getByText("shapez.io")).toBeInTheDocument();
  });

  it("renders raw name for unknown game", () => {
    render(<GameBadge game="tetris" />);
    expect(screen.getByText("tetris")).toBeInTheDocument();
  });

  it("applies game-specific color class", () => {
    const { container } = render(<GameBadge game="breakout71" />);
    const badge = container.querySelector("span");
    expect(badge?.className).toContain("text-violet-400");
  });
});
