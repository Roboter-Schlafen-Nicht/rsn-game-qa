import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SeverityBadge } from "@/components/SeverityBadge";

describe("SeverityBadge", () => {
  it("renders severity text", () => {
    render(<SeverityBadge severity="critical" />);
    expect(screen.getByText("critical")).toBeInTheDocument();
  });

  it("renders count when provided", () => {
    render(<SeverityBadge severity="warning" count={7} />);
    expect(screen.getByText("warning")).toBeInTheDocument();
    expect(screen.getByText("(7)")).toBeInTheDocument();
  });

  it("does not render count when not provided", () => {
    const { container } = render(<SeverityBadge severity="info" />);
    expect(container.querySelector(".font-mono")).toBeNull();
  });

  it("renders count of zero", () => {
    render(<SeverityBadge severity="info" count={0} />);
    expect(screen.getByText("(0)")).toBeInTheDocument();
  });

  it("applies red dot for critical severity", () => {
    const { container } = render(<SeverityBadge severity="critical" />);
    const dot = container.querySelector(".bg-red-400");
    expect(dot).toBeInTheDocument();
  });

  it("applies amber dot for warning severity", () => {
    const { container } = render(<SeverityBadge severity="warning" />);
    const dot = container.querySelector(".bg-amber-400");
    expect(dot).toBeInTheDocument();
  });

  it("applies blue dot for info severity", () => {
    const { container } = render(<SeverityBadge severity="info" />);
    const dot = container.querySelector(".bg-blue-400");
    expect(dot).toBeInTheDocument();
  });
});
