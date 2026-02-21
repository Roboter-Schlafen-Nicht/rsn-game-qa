import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { Layout } from "@/components/Layout";

function renderLayout(initialRoute = "/overview") {
  return render(
    <MemoryRouter initialEntries={[initialRoute]}>
      <Layout />
    </MemoryRouter>,
  );
}

describe("Layout", () => {
  it("renders the logo text", () => {
    renderLayout();
    expect(screen.getByText("RSN Game QA")).toBeInTheDocument();
  });

  it("renders all navigation items", () => {
    renderLayout();
    expect(screen.getByText("Overview")).toBeInTheDocument();
    expect(screen.getByText("Training")).toBeInTheDocument();
    expect(screen.getByText("Evaluation")).toBeInTheDocument();
    expect(screen.getByText("Findings")).toBeInTheDocument();
    expect(screen.getByText("Cross-Game")).toBeInTheDocument();
  });

  it("renders navigation links with correct hrefs", () => {
    renderLayout();
    const links = screen.getAllByRole("link");
    const hrefs = links.map((l) => l.getAttribute("href"));
    expect(hrefs).toContain("/overview");
    expect(hrefs).toContain("/training");
    expect(hrefs).toContain("/evaluation");
    expect(hrefs).toContain("/findings");
    expect(hrefs).toContain("/cross-game");
  });

  it("renders version footer", () => {
    renderLayout();
    expect(screen.getByText(/v0\.1\.0-alpha/)).toBeInTheDocument();
  });
});
