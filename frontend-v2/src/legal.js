const LEGAL_COPY = {
  title: "Responsible Use Notice",
  intro:
    "Use Oh Sheet only with content you are legally allowed to access, upload, convert, download, and share.",
  bullets: [
    "You are responsible for complying with copyright law, licenses, platform terms, and any permissions required for the audio, video, MIDI, sheet music, or other material you use with this app.",
    "Do not use Oh Sheet to infringe intellectual property rights, misuse third-party content, bypass restrictions, or otherwise violate applicable law or terms of service.",
    "AI-generated transcriptions, arrangements, and sheet music may contain errors or omissions, so review all outputs before relying on or distributing them.",
  ],
  acknowledgement:
    "By continuing, you acknowledge that you are solely responsible for your use of Oh Sheet and any content you submit or generate. Oh Sheet is provided as-is, and Oh Sheet takes no responsibility for improper, unauthorized, or unlawful use.",
};

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value == null) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    }
  }
  for (const child of children) appendChild(node, child);
  return node;
}

function appendChild(parent, child) {
  if (child == null || child === false) return;
  if (Array.isArray(child)) {
    for (const nested of child) appendChild(parent, nested);
    return;
  }
  if (typeof child === "string") {
    parent.appendChild(document.createTextNode(child));
    return;
  }
  parent.appendChild(child);
}

function icon(name) {
  return el("span", { class: "material-symbols-rounded", text: name });
}

export function mountLegalDisclaimer({ root = document.body } = {}) {
  const previouslyFocused = document.activeElement;
  const titleId = "legal-modal-title";
  const copyId = "legal-modal-copy";

  const dismissBtn = el(
    "button",
    {
      class: "legal-dismiss",
      type: "button",
      "aria-label": "Dismiss responsible use notice",
    },
    icon("close"),
  );
  const continueBtn = el(
    "button",
    { class: "btn-filled legal-continue", type: "button" },
    icon("verified_user"),
    "Continue responsibly",
  );
  const panel = el(
    "section",
    {
      class: "legal-modal",
      role: "dialog",
      "aria-modal": "true",
      "aria-labelledby": titleId,
      "aria-describedby": copyId,
    },
    dismissBtn,
    el(
      "div",
      { class: "legal-badge" },
      icon("gavel"),
      el("span", { text: "Legal" }),
    ),
    el("h2", { class: "legal-title", id: titleId, text: LEGAL_COPY.title }),
    el("p", { class: "legal-intro", id: copyId, text: LEGAL_COPY.intro }),
    el(
      "ul",
      { class: "legal-list" },
      LEGAL_COPY.bullets.map((bullet) => el("li", { text: bullet })),
    ),
    el("p", { class: "legal-acknowledgement", text: LEGAL_COPY.acknowledgement }),
    el(
      "p",
      {
        class: "legal-note",
        text: "If you are unsure whether you have permission to use specific content, do not upload it.",
      },
    ),
    el("div", { class: "actions legal-actions" }, continueBtn),
  );
  const backdrop = el("div", { class: "legal-backdrop" }, panel);

  let closed = false;
  function close() {
    if (closed) return;
    closed = true;
    backdrop.remove();
    document.body.classList.remove("legal-modal-open");
    document.removeEventListener("keydown", onKeyDown);
    if (previouslyFocused && typeof previouslyFocused.focus === "function") {
      previouslyFocused.focus();
    }
  }

  function onKeyDown(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      close();
    }
  }

  dismissBtn.addEventListener("click", close);
  continueBtn.addEventListener("click", close);
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) close();
  });

  root.appendChild(backdrop);
  document.body.classList.add("legal-modal-open");
  document.addEventListener("keydown", onKeyDown);
  continueBtn.focus();

  return {
    backdrop,
    panel,
    close,
  };
}

export { LEGAL_COPY };
