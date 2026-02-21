import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { FindingsPage } from "@/pages/Findings";
import { MOCK_SESSIONS } from "@/data/mock";
import { ORACLE_INFO } from "@/types";
import { formatNumber } from "@/lib/utils";

function renderPage() {
  return render(
    <MemoryRouter>
      <FindingsPage />
    </MemoryRouter>,
  );
}

describe("FindingsPage", () => {
  const totalFromSummaries = MOCK_SESSIONS.reduce(
    (sum, s) => sum + s.summary.total_findings,
    0,
  );
  const criticalFromSummaries = MOCK_SESSIONS.reduce(
    (sum, s) => sum + s.summary.critical_findings,
    0,
  );
  const warningFromSummaries = MOCK_SESSIONS.reduce(
    (sum, s) => sum + s.summary.warning_findings,
    0,
  );
  const infoFromSummaries = MOCK_SESSIONS.reduce(
    (sum, s) => sum + s.summary.info_findings,
    0,
  );

  it("renders the page heading", () => {
    renderPage();
    expect(screen.getByText("Findings")).toBeInTheDocument();
    expect(
      screen.getByText("Oracle findings across all evaluation sessions"),
    ).toBeInTheDocument();
  });

  it("shows total findings stat card", () => {
    renderPage();
    expect(screen.getByText("Total Findings")).toBeInTheDocument();
    expect(
      screen.getByText(formatNumber(totalFromSummaries)),
    ).toBeInTheDocument();
  });

  it("shows critical findings stat card", () => {
    renderPage();
    // The stat card label
    expect(screen.getByText("Critical")).toBeInTheDocument();
    // The formatted count may appear in both the stat card and oracle count spans
    expect(
      screen.getAllByText(formatNumber(criticalFromSummaries)).length,
    ).toBeGreaterThanOrEqual(1);
  });

  it("shows warnings stat card", () => {
    renderPage();
    expect(screen.getByText("Warnings")).toBeInTheDocument();
    expect(
      screen.getByText(formatNumber(warningFromSummaries)),
    ).toBeInTheDocument();
  });

  it("shows info stat card", () => {
    renderPage();
    expect(screen.getByText("Info")).toBeInTheDocument();
    expect(
      screen.getByText(formatNumber(infoFromSummaries)),
    ).toBeInTheDocument();
  });

  it("renders findings by oracle section", () => {
    renderPage();
    expect(screen.getByText("Findings by Oracle")).toBeInTheDocument();
  });

  it("shows oracle labels for oracles that have findings", () => {
    renderPage();
    const allFindings = MOCK_SESSIONS.flatMap((s) =>
      s.episodes.flatMap((ep) => ep.findings),
    );
    const oracleNames = [...new Set(allFindings.map((f) => f.oracle_name))];
    for (const name of oracleNames) {
      const info = ORACLE_INFO[name as keyof typeof ORACLE_INFO];
      if (info) {
        expect(screen.getByText(info.label)).toBeInTheDocument();
      }
    }
  });

  it("renders critical findings detail section when critical findings exist", () => {
    renderPage();
    const allFindings = MOCK_SESSIONS.flatMap((s) =>
      s.episodes.flatMap((ep) => ep.findings),
    );
    const critical = allFindings.filter((f) => f.severity === "critical");
    if (critical.length > 0) {
      expect(
        screen.getByText(`Critical Findings Detail (${critical.length})`),
      ).toBeInTheDocument();
    }
  });

  it("shows critical finding descriptions", () => {
    renderPage();
    const criticalFindings = MOCK_SESSIONS.flatMap((s) =>
      s.episodes.flatMap((ep) =>
        ep.findings.filter((f) => f.severity === "critical"),
      ),
    );
    for (const f of criticalFindings) {
      // Each critical finding description appears in both the "by oracle" list
      // and the detail section, so we check getAllByText
      expect(screen.getAllByText(f.description).length).toBeGreaterThanOrEqual(1);
    }
  });

  it("shows JSON data for critical findings with non-empty data", () => {
    renderPage();
    const criticalFindings = MOCK_SESSIONS.flatMap((s) =>
      s.episodes.flatMap((ep) =>
        ep.findings.filter(
          (f) => f.severity === "critical" && Object.keys(f.data).length > 0,
        ),
      ),
    );
    // JSON is rendered in <pre> blocks — query them directly since
    // Testing Library normalizes whitespace in getByText
    const preElements = document.querySelectorAll("pre");
    for (const f of criticalFindings) {
      const jsonText = JSON.stringify(f.data, null, 2);
      const found = Array.from(preElements).some(
        (el) => el.textContent === jsonText,
      );
      expect(found).toBe(true);
    }
  });

  it("renders warnings section when warning findings exist", () => {
    renderPage();
    const allFindings = MOCK_SESSIONS.flatMap((s) =>
      s.episodes.flatMap((ep) => ep.findings),
    );
    const warnings = allFindings.filter((f) => f.severity === "warning");
    if (warnings.length > 0) {
      expect(
        screen.getByText(`Warnings (${warnings.length})`),
      ).toBeInTheDocument();
    }
  });

  it("renders info section when info findings exist", () => {
    renderPage();
    const allFindings = MOCK_SESSIONS.flatMap((s) =>
      s.episodes.flatMap((ep) => ep.findings),
    );
    const info = allFindings.filter((f) => f.severity === "info");
    if (info.length > 0) {
      expect(
        screen.getByText(`Info (${info.length})`),
      ).toBeInTheDocument();
    }
  });
});
