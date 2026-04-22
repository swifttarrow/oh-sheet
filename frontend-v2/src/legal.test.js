import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { LEGAL_COPY, mountLegalDisclaimer } from "./legal.js";

describe("mountLegalDisclaimer", () => {
  let root;

  beforeEach(() => {
    root = document.createElement("div");
    document.body.appendChild(root);
  });

  afterEach(() => {
    document.body.classList.remove("legal-modal-open");
    root.remove();
    document.querySelectorAll(".legal-backdrop").forEach((node) => node.remove());
  });

  it("renders the responsible use copy as a modal dialog", () => {
    const { panel } = mountLegalDisclaimer({ root });

    expect(panel.getAttribute("role")).toBe("dialog");
    expect(panel.textContent).toContain(LEGAL_COPY.title);
    expect(panel.textContent).toContain("Oh Sheet takes no responsibility");
    expect(document.body.classList.contains("legal-modal-open")).toBe(true);
  });

  it("dismisses when the primary continue button is clicked", () => {
    const { backdrop } = mountLegalDisclaimer({ root });

    backdrop.querySelector(".legal-continue").click();

    expect(root.querySelector(".legal-backdrop")).toBeNull();
    expect(document.body.classList.contains("legal-modal-open")).toBe(false);
  });

  it("dismisses when escape is pressed", () => {
    mountLegalDisclaimer({ root });

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));

    expect(root.querySelector(".legal-backdrop")).toBeNull();
  });

  it("dismisses when the backdrop is clicked", () => {
    const { backdrop, panel } = mountLegalDisclaimer({ root });

    panel.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(root.querySelector(".legal-backdrop")).not.toBeNull();

    backdrop.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(root.querySelector(".legal-backdrop")).toBeNull();
  });
});
