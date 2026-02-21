import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { FindingsTable } from "@/components/FindingsTable";
import type { FindingReport } from "@/types";

function makeFinding(overrides: Partial<FindingReport> = {}): FindingReport {
  return {
    oracle_name: "crash",
    severity: "info",
    step: 100,
    description: "Test finding",
    data: {},
    screenshot_path: null,
    ...overrides,
  };
}

describe("FindingsTable", () => {
  it("renders empty state when no findings", () => {
    render(<FindingsTable findings={[]} />);
    expect(screen.getByText("No findings recorded")).toBeInTheDocument();
  });

  it("renders findings with oracle name, step, and description", () => {
    const findings = [
      makeFinding({ oracle_name: "stuck", step: 42, description: "Frame frozen for 5s" }),
    ];
    render(<FindingsTable findings={findings} />);
    expect(screen.getByText("stuck")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("Frame frozen for 5s")).toBeInTheDocument();
  });

  it("renders table headers", () => {
    const findings = [makeFinding()];
    render(<FindingsTable findings={findings} />);
    expect(screen.getByText("Severity")).toBeInTheDocument();
    expect(screen.getByText("Oracle")).toBeInTheDocument();
    expect(screen.getByText("Step")).toBeInTheDocument();
    expect(screen.getByText("Description")).toBeInTheDocument();
  });

  it("sorts findings by severity (critical first, then warning, then info)", () => {
    const findings = [
      makeFinding({ severity: "info", step: 10, oracle_name: "info_oracle" }),
      makeFinding({ severity: "critical", step: 20, oracle_name: "critical_oracle" }),
      makeFinding({ severity: "warning", step: 5, oracle_name: "warning_oracle" }),
    ];
    render(<FindingsTable findings={findings} />);

    const oracleNames = screen.getAllByText(/oracle/).map((el) => el.textContent);
    expect(oracleNames).toEqual(["critical_oracle", "warning_oracle", "info_oracle"]);
  });

  it("sorts same-severity findings by step", () => {
    const findings = [
      makeFinding({ severity: "warning", step: 300, oracle_name: "later" }),
      makeFinding({ severity: "warning", step: 100, oracle_name: "earlier" }),
    ];
    render(<FindingsTable findings={findings} />);

    const oracleNames = screen.getAllByText(/earlier|later/).map((el) => el.textContent);
    expect(oracleNames).toEqual(["earlier", "later"]);
  });

  it("limits displayed rows with maxRows", () => {
    const findings = [
      makeFinding({ step: 1 }),
      makeFinding({ step: 2 }),
      makeFinding({ step: 3 }),
    ];
    render(<FindingsTable findings={findings} maxRows={2} />);

    // Should show "Showing 2 of 3" footer
    expect(screen.getByText(/Showing 2 of 3/)).toBeInTheDocument();
  });

  it("does not show footer when all rows fit", () => {
    const findings = [makeFinding({ step: 1 })];
    render(<FindingsTable findings={findings} maxRows={5} />);

    expect(screen.queryByText(/Showing/)).toBeNull();
  });

  it("renders severity badges for each finding", () => {
    const findings = [
      makeFinding({ severity: "critical" }),
      makeFinding({ severity: "warning", step: 200 }),
    ];
    render(<FindingsTable findings={findings} />);

    expect(screen.getByText("critical")).toBeInTheDocument();
    expect(screen.getByText("warning")).toBeInTheDocument();
  });
});
