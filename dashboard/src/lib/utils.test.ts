import { describe, it, expect } from "vitest";
import {
  formatNumber,
  formatDuration,
  severityColor,
  severityBgColor,
  gameDisplayName,
  gameColor,
  gameChartColor,
} from "@/lib/utils";

describe("formatNumber", () => {
  it("formats integers with no decimals by default", () => {
    expect(formatNumber(1000)).toBe("1,000");
  });

  it("formats large numbers with separators", () => {
    expect(formatNumber(184314)).toBe("184,314");
  });

  it("formats with specified decimal places", () => {
    expect(formatNumber(3.14159, 2)).toBe("3.14");
  });

  it("adds trailing zeros when decimals specified", () => {
    expect(formatNumber(5, 2)).toBe("5.00");
  });

  it("handles zero", () => {
    expect(formatNumber(0)).toBe("0");
  });

  it("handles negative numbers", () => {
    expect(formatNumber(-2048)).toBe("-2,048");
  });
});

describe("formatDuration", () => {
  it("formats sub-minute as seconds", () => {
    expect(formatDuration(45.3)).toBe("45.3s");
  });

  it("formats exactly 60 seconds as minutes", () => {
    expect(formatDuration(60)).toBe("1m 0s");
  });

  it("formats minutes and seconds", () => {
    expect(formatDuration(185)).toBe("3m 5s");
  });

  it("formats hours and minutes", () => {
    expect(formatDuration(3660)).toBe("1h 1m");
  });

  it("formats exactly one hour", () => {
    expect(formatDuration(3600)).toBe("1h 0m");
  });

  it("handles zero seconds", () => {
    expect(formatDuration(0)).toBe("0.0s");
  });

  it("handles sub-second durations", () => {
    expect(formatDuration(0.5)).toBe("0.5s");
  });
});

describe("severityColor", () => {
  it("returns red for critical", () => {
    expect(severityColor("critical")).toBe("text-red-400");
  });

  it("returns amber for warning", () => {
    expect(severityColor("warning")).toBe("text-amber-400");
  });

  it("returns blue for info", () => {
    expect(severityColor("info")).toBe("text-blue-400");
  });
});

describe("severityBgColor", () => {
  it("returns red classes for critical", () => {
    const result = severityBgColor("critical");
    expect(result).toContain("bg-red-500/10");
    expect(result).toContain("text-red-400");
    expect(result).toContain("border-red-500/20");
  });

  it("returns amber classes for warning", () => {
    const result = severityBgColor("warning");
    expect(result).toContain("bg-amber-500/10");
    expect(result).toContain("text-amber-400");
  });

  it("returns blue classes for info", () => {
    const result = severityBgColor("info");
    expect(result).toContain("bg-blue-500/10");
    expect(result).toContain("text-blue-400");
  });
});

describe("gameDisplayName", () => {
  it("maps breakout71 to Breakout 71", () => {
    expect(gameDisplayName("breakout71")).toBe("Breakout 71");
  });

  it("maps hextris to Hextris", () => {
    expect(gameDisplayName("hextris")).toBe("Hextris");
  });

  it("maps shapez to shapez.io", () => {
    expect(gameDisplayName("shapez")).toBe("shapez.io");
  });

  it("returns raw name for unknown games", () => {
    expect(gameDisplayName("tetris")).toBe("tetris");
  });
});

describe("gameColor", () => {
  it("returns violet for breakout71", () => {
    expect(gameColor("breakout71")).toBe("text-violet-400");
  });

  it("returns cyan for hextris", () => {
    expect(gameColor("hextris")).toBe("text-cyan-400");
  });

  it("returns emerald for shapez", () => {
    expect(gameColor("shapez")).toBe("text-emerald-400");
  });

  it("returns zinc for unknown games", () => {
    expect(gameColor("unknown")).toBe("text-zinc-400");
  });
});

describe("gameChartColor", () => {
  it("returns purple hex for breakout71", () => {
    expect(gameChartColor("breakout71")).toBe("#8b5cf6");
  });

  it("returns cyan hex for hextris", () => {
    expect(gameChartColor("hextris")).toBe("#06b6d4");
  });

  it("returns emerald hex for shapez", () => {
    expect(gameChartColor("shapez")).toBe("#10b981");
  });

  it("returns zinc hex for unknown games", () => {
    expect(gameChartColor("unknown")).toBe("#71717a");
  });
});
