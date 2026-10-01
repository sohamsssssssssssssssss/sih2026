import { render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";

vi.mock("gsap", () => ({
  default: {
    context: () => ({ revert: vi.fn() }),
    fromTo: vi.fn(),
    registerPlugin: vi.fn(),
  },
}));
vi.mock("gsap/ScrollTrigger", () => ({ ScrollTrigger: {} }));

import Services from "./Services";

test("keeps decorative typography clipped outside the service content layer", () => {
  const { container } = render(<Services />);
  const typeRegion = container.querySelector(".services-type-region");
  const content = container.querySelector(".z-20");

  expect(typeRegion?.classList.contains("overflow-hidden")).toBe(true);
  expect(content?.classList.contains("bg-white")).toBe(true);
  expect(typeRegion?.contains(screen.getByText("Geospatial VQA"))).toBe(false);
});
