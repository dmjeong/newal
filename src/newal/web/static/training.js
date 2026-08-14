"use strict";

// Training mode. Separate page, separate address, no chat here on purpose:
// the two modes have different risks and different audiences.

const $ = (id) => document.getElementById(id);
const form = $("plan-form");

let summary = null;
let script = "";
let scriptName = "train_sft.py";

// ---- summary ---------------------------------------------------------------

async function loadSummary() {
  const response = await fetch("/api/training/summary");
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    $("counts").tBodies[0].replaceChildren(row("오류", detail.detail || response.status));
    return;
  }
  summary = await response.json();

  $("capture-warning").hidden = summary.capture_enabled;
  $("db-path").textContent = summary.db_path;

  $("counts").tBodies[0].replaceChildren(
    ...Object.entries(summary.counts).map(([key, value]) =>
      row(captureLabel(key), value.toLocaleString())
    )
  );

  $("model-options").replaceChildren(
    ...summary.candidate_models.map((id) => {
      const option = document.createElement("option");
      option.value = id;
      return option;
    })
  );

  fillForm({ ...summary.defaults, base_model: summary.suggested_base_model });

  for (const fmt of summary.formats) {
    $(`dl-${fmt}`).href = `/api/training/dataset/${fmt}`;
  }
}

function row(label, value) {
  const tr = document.createElement("tr");
  const a = document.createElement("td");
  a.textContent = label;
  const b = document.createElement("td");
  b.textContent = value;
  tr.append(a, b);
  return tr;
}

// ---- the form --------------------------------------------------------------

function fillForm(values) {
  for (const [key, value] of Object.entries(values)) {
    const field = form.elements[key];
    if (!field) continue;
    field.value = Array.isArray(value) ? value.join(", ") : value;
  }
  syncTaskFields();
  showEffective();
}

function readForm() {
  const data = new FormData(form);
  const plan = {};
  for (const [key, raw] of data.entries()) {
    if (key === "target_modules") {
      plan[key] = raw.split(",").map((s) => s.trim()).filter(Boolean);
    } else if (["task", "base_model", "method", "output_dir"].includes(key)) {
      plan[key] = raw;
    } else {
      plan[key] = Number(raw);
    }
  }
  return plan;
}

function syncTaskFields() {
  const isDpo = form.elements.task.value === "dpo";
  $("beta-field").style.opacity = isDpo ? "1" : "0.4";
  form.elements.beta.disabled = !isDpo;

  const isLora = form.elements.method.value !== "full";
  for (const name of ["lora_r", "lora_alpha", "lora_dropout", "target_modules"]) {
    form.elements[name].disabled = !isLora;
  }
}

function showEffective() {
  const batch = Number(form.elements.batch_size.value) || 1;
  const accum = Number(form.elements.grad_accum.value) || 1;
  $("effective").textContent = `실효 배치 ${batch * accum}`;
}

form.addEventListener("input", () => {
  syncTaskFields();
  showEffective();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const plan = readForm();

  const response = await fetch("/api/training/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(plan),
  });

  const warnings = $("plan-warnings");
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    warnings.replaceChildren(notice(detail.detail || `HTTP ${response.status}`, "bad"));
    $("script-section").hidden = true;
    return;
  }

  const result = await response.json();
  script = result.script;
  scriptName = result.script_name;

  warnings.replaceChildren(
    ...result.warnings.map((text) => notice(text, "warn")),
    notice(
      `${result.samples.toLocaleString()}개 샘플이 ${result.dataset} 로 나갑니다.`,
      result.samples > 0 ? "" : "warn"
    )
  );

  $("script").textContent = script;
  $("script-meta").textContent = `${scriptName} · 데이터셋 ${result.dataset}`;
  const blob = new Blob([script], { type: "text/x-python" });
  $("dl-script").href = URL.createObjectURL(blob);
  $("dl-script").download = scriptName;
  $("script-section").hidden = false;
  $("script-section").scrollIntoView({ behavior: "smooth", block: "start" });
});

function notice(text, kind = "") {
  const div = document.createElement("div");
  div.className = `notice ${kind}`.trim();
  div.textContent = text;
  return div;
}

// ---- actions ---------------------------------------------------------------

$("copy-script").addEventListener("click", async () => {
  await navigator.clipboard.writeText(script);
  $("copy-script").textContent = "복사됨";
  setTimeout(() => ($("copy-script").textContent = "복사"), 1500);
});

$("clear").addEventListener("click", async () => {
  const total = (summary?.counts?.turns || 0) + (summary?.counts?.repair_pairs || 0);
  if (!total) return;
  if (!confirm(`수집된 ${total.toLocaleString()}건을 전부 지울까요? 되돌릴 수 없습니다.`)) {
    return;
  }
  await fetch("/api/training/clear", { method: "POST" });
  loadSummary();
});

// Downloads 404 when nothing has been captured; say so instead of a blank tab.
for (const fmt of ["sft", "dpo", "routing"]) {
  $(`dl-${fmt}`).addEventListener("click", async (event) => {
    const probe = await fetch(`/api/training/dataset/${fmt}`, { method: "HEAD" });
    if (probe.status === 404) {
      event.preventDefault();
      $("plan-warnings").replaceChildren(
        notice(`${fmt.toUpperCase()} 샘플이 아직 없습니다. newal을 더 쓰면 쌓입니다.`, "warn")
      );
    }
  });
}

fetch("/api/state")
  .then((r) => r.json())
  .then((s) => ($("version").textContent = "v" + s.version))
  .catch(() => {});

loadSummary();
