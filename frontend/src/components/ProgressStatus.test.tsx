import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { JobSnapshot } from "../types";
import { ProgressStatus } from "./ProgressStatus";

it("announces measured progress and terminal errors", () => {
  const job = {
    state: "failed",
    message: "Processing failed",
    progress: 0.42,
    error: "GPU allocation failed",
  } as JobSnapshot;

  render(<ProgressStatus job={job} />);

  const status = screen.getByRole("status");
  expect(status.getAttribute("aria-live")).toBe("polite");
  expect(status.textContent).toContain("42%");
  expect(status.textContent).toContain("GPU allocation failed");
});
