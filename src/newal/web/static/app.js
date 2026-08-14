"use strict";

// No framework and no build step: this file is served as-is, so `pip install`
// is the whole toolchain. The server streams server-sent events; everything
// here is reading that stream and appending to the DOM.

const $ = (id) => document.getElementById(id);
const messages = $("messages");
const input = $("input");
const form = $("form");
const send = $("send");

let attachments = [];
let streaming = false;

// ---- state panel -----------------------------------------------------------

async function refreshState() {
  let state;
  try {
    state = await (await fetch("/api/state")).json();
  } catch {
    return; // server restarting; the next poll picks it up
  }

  $("version").textContent = "v" + state.version;
  $("workspace").textContent = state.workspace;

  $("models").replaceChildren(
    ...state.models.map((line) => {
      const li = document.createElement("li");
      li.textContent = line;
      return li;
    })
  );

  $("routing").textContent =
    `${state.router.strategy} · ${state.router.mode}` +
    (state.router.classifier ? " · 분류기 활성" : " · 휴리스틱만");

  fillTable("usage", Object.entries(state.usage).map(([k, v]) => [k, v.toLocaleString()]));
  fillTable(
    "captured",
    Object.entries(state.captured || {}).map(([k, v]) => [k.replace(/_/g, " "), v])
  );

  const policy = { ask: "물어봄", allow: "자동 허용", deny: "차단" }[state.shell_policy];
  $("shell-policy").textContent = `셸 실행: ${policy}`;
}

function fillTable(id, rows) {
  const body = $(id).tBodies[0];
  body.replaceChildren(
    ...rows.map(([label, value]) => {
      const tr = document.createElement("tr");
      const a = document.createElement("td");
      a.textContent = label;
      const b = document.createElement("td");
      b.textContent = value;
      tr.append(a, b);
      return tr;
    })
  );
}

// ---- attachments -----------------------------------------------------------

async function uploadFiles(files) {
  for (const file of files) {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/api/upload", { method: "POST", body });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      addSystemLine(`업로드 실패: ${detail.detail || response.status}`, "error");
      continue;
    }
    attachments.push(await response.json());
  }
  renderAttachments();
}

function renderAttachments() {
  const box = $("attachments");
  box.hidden = attachments.length === 0;
  box.replaceChildren(
    ...attachments.map((item) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.append(`${item.kind === "video" ? "🎬" : "🖼"} ${item.name}`);

      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.title = "제거";
      remove.onclick = async () => {
        await fetch(`/api/upload/${item.id}`, { method: "DELETE" });
        attachments = attachments.filter((a) => a.id !== item.id);
        renderAttachments();
      };
      chip.append(remove);
      return chip;
    })
  );
}

// ---- rendering -------------------------------------------------------------

function addTurn(role) {
  const turn = document.createElement("div");
  turn.className = `turn ${role}`;
  messages.append(turn);
  return turn;
}

function addBubble(turn, text) {
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  turn.append(bubble);
  scroll();
  return bubble;
}

function addTraceLine(turn, kind, text) {
  let trace = turn.querySelector(".trace");
  if (!trace) {
    trace = document.createElement("div");
    trace.className = "trace";
    turn.append(trace);
  }
  const line = document.createElement("div");
  line.className = `line ${kind}`;
  line.textContent = text;
  trace.append(line);
  scroll();
}

function addSystemLine(text, kind = "error") {
  const turn = addTurn("assistant");
  addTraceLine(turn, kind, text);
}

function scroll() {
  messages.scrollTop = messages.scrollHeight;
}

// ---- approvals -------------------------------------------------------------

function askApproval(payload) {
  const dialog = $("approval");
  $("approval-detail").textContent = payload.detail;
  dialog.showModal();

  const answer = async (allow) => {
    dialog.close();
    const body = new FormData();
    body.append("request_id", payload.id);
    body.append("allow", allow ? "true" : "false");
    await fetch("/api/approve", { method: "POST", body });
  };
  $("allow").onclick = () => answer(true);
  $("deny").onclick = () => answer(false);
  // Esc closes the dialog; treat that as a refusal rather than leaving the
  // agent's worker thread blocked until it times out.
  dialog.oncancel = () => answer(false);
}

// ---- the turn --------------------------------------------------------------

async function sendMessage(text) {
  streaming = true;
  send.disabled = true;

  addBubble(addTurn("user"), text);
  const turn = addTurn("assistant");
  const pending = document.createElement("span");
  pending.className = "spinner";
  turn.append(pending);

  const body = new FormData();
  body.append("message", text);

  let response;
  try {
    response = await fetch("/api/chat", { method: "POST", body });
  } catch (error) {
    pending.remove();
    addTraceLine(turn, "error", `서버에 연결하지 못했습니다: ${error}`);
    return finish();
  }
  if (!response.ok) {
    pending.remove();
    const detail = await response.json().catch(() => ({}));
    addTraceLine(turn, "error", detail.detail || `HTTP ${response.status}`);
    return finish();
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, boundary).replace(/^data: /, "");
      buffer = buffer.slice(boundary + 2);
      if (!raw.trim()) continue;
      handleEvent(turn, pending, JSON.parse(raw));
    }
  }
  finish();
}

function handleEvent(turn, pending, event) {
  const { kind, payload } = event;

  if (kind === "approval") return askApproval(payload);
  if (kind === "end") return;

  if (kind === "done") {
    pending.remove();
    // The loop narrates its final prose as an `assistant` event and then
    // repeats it in `done`. Drop the narration so the answer appears once.
    if (payload.text) {
      for (const line of turn.querySelectorAll(".assistant-note")) {
        if (line.textContent === payload.text) line.remove();
      }
      addBubble(turn, payload.text);
    }
    addSummary(turn, payload);
    refreshState();
    return;
  }
  if (kind === "error") {
    pending.remove();
    return addTraceLine(turn, "error", payload.message);
  }
  if (kind === "assistant") {
    // Interim prose while the agent is still using tools.
    return addTraceLine(turn, "assistant-note", payload.text);
  }
  if (kind === "attachment") {
    return addTraceLine(turn, "tool_result", `첨부: ${payload.summary}`);
  }
  addTraceLine(turn, kind, payload.text ?? "");
}

function addSummary(turn, payload) {
  const parts = [];
  if (payload.routes?.length) parts.push(`${payload.routes.length}회 호출`);
  if (payload.escalations) parts.push(`에스컬레이션 ${payload.escalations}회`);
  if (payload.tokens) parts.push(`${payload.tokens.toLocaleString()} 토큰`);
  if (payload.files_written?.length) parts.push(`변경: ${payload.files_written.join(", ")}`);

  const summary = document.createElement("div");
  summary.className = "summary";
  for (const part of parts) {
    const span = document.createElement("span");
    span.textContent = part;
    summary.append(span);
  }
  if (payload.verification) {
    const span = document.createElement("span");
    span.className = payload.verification.passed ? "pass" : "fail";
    span.textContent =
      `검증 (${payload.verification.command}): ` +
      (payload.verification.passed ? "통과" : "실패");
    summary.append(span);
  }
  if (summary.childElementCount) turn.append(summary);
  scroll();
}

function finish() {
  streaming = false;
  send.disabled = false;
  attachments = [];
  renderAttachments();
  input.focus();
}

// ---- wiring ----------------------------------------------------------------

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || streaming) return;
  input.value = "";
  resize();
  sendMessage(text);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

function resize() {
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
}
input.addEventListener("input", resize);

$("file").addEventListener("change", (event) => {
  uploadFiles(event.target.files);
  event.target.value = "";
});

let dragDepth = 0;
document.addEventListener("dragenter", (event) => {
  event.preventDefault();
  if (++dragDepth === 1) {
    document.body.classList.add("dragging");
    $("drop-hint").hidden = false;
  }
});
document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) endDrag();
});
document.addEventListener("drop", (event) => {
  event.preventDefault();
  endDrag();
  if (event.dataTransfer?.files?.length) uploadFiles(event.dataTransfer.files);
});
function endDrag() {
  dragDepth = 0;
  document.body.classList.remove("dragging");
  $("drop-hint").hidden = true;
}

$("reset").addEventListener("click", async () => {
  if (streaming) return;
  await fetch("/api/reset", { method: "POST" });
  messages.replaceChildren();
  attachments = [];
  renderAttachments();
  refreshState();
});

refreshState();
setInterval(refreshState, 15000);
