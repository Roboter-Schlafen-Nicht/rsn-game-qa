import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatCard } from "@/components/StatCard";
import { Bug } from "lucide-react";

describe("StatCard", () => {
  it("renders label and value", () => {
    render(<StatCard label="Total Findings" value={42} icon={Bug} />);
    expect(screen.getByText("Total Findings")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("renders string value", () => {
    render(<StatCard label="Score" value="1,234" icon={Bug} />);
    expect(screen.getByText("1,234")).toBeInTheDocument();
  });

  it("renders subtitle when provided", () => {
    render(
      <StatCard label="Games" value={2} icon={Bug} subtitle="breakout71, hextris" />,
    );
    expect(screen.getByText("breakout71, hextris")).toBeInTheDocument();
  });

  it("does not render subtitle when not provided", () => {
    const { container } = render(
      <StatCard label="Games" value={2} icon={Bug} />,
    );
    // subtitle element should not exist
    const subtitleEl = container.querySelector(".text-xs.text-zinc-500");
    expect(subtitleEl).toBeNull();
  });

  it("renders positive trend with + prefix", () => {
    render(
      <StatCard
        label="Coverage"
        value={98}
        icon={Bug}
        trend={{ value: 12, label: "vs baseline" }}
      />,
    );
    expect(screen.getByText("+12% vs baseline")).toBeInTheDocument();
  });

  it("renders negative trend without + prefix", () => {
    render(
      <StatCard
        label="Reward"
        value={-2.48}
        icon={Bug}
        trend={{ value: -34, label: "vs random" }}
      />,
    );
    expect(screen.getByText("-34% vs random")).toBeInTheDocument();
  });

  it("applies custom color class to value", () => {
    render(
      <StatCard label="Critical" value={4} icon={Bug} color="text-red-400" />,
    );
    const value = screen.getByText("4");
    expect(value.className).toContain("text-red-400");
  });

  it("renders children", () => {
    render(
      <StatCard label="Test" value={1} icon={Bug}>
        <div data-testid="child">Child content</div>
      </StatCard>,
    );
    expect(screen.getByTestId("child")).toBeInTheDocument();
  });
});
