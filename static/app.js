/**
 * Hiver AI Customer Support Copilot — Frontend Application Logic
 */

// State
let presetsData = [];

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  loadPresets();
  loadMetrics();
  setupTextareaCounter();
});

// Theme Management (Light / Dark)
function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved === "light") {
    document.documentElement.classList.remove("dark");
    updateThemeBtn(false);
  } else {
    document.documentElement.classList.add("dark");
    updateThemeBtn(true);
  }
}

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle("dark");
  localStorage.setItem("theme", isDark ? "dark" : "light");
  updateThemeBtn(isDark);
}

function updateThemeBtn(isDark) {
  const btn = document.getElementById("theme-toggle-btn");
  if (btn) {
    if (isDark) {
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg><span>Dark</span>`;
    } else {
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg><span>Light</span>`;
    }
  }
}

// Tab switching
function switchTab(tabId) {
  document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));

  const targetPane = document.getElementById(`tab-${tabId}`);
  const targetBtn = document.getElementById(`tab-${tabId}-btn`);

  if (targetPane) targetPane.classList.add("active");
  if (targetBtn) targetBtn.classList.add("active");
}

// Textarea Character Counter
function setupTextareaCounter() {
  const input = document.getElementById("query-input");
  const counter = document.getElementById("char-count");

  input.addEventListener("input", () => {
    const len = input.value.length;
    counter.textContent = `${len} / 280 chars`;
    if (len > 280) {
      counter.style.color = "#ef4444";
    } else {
      counter.style.color = "var(--muted-foreground)";
    }
  });
}

// Fetch and Render Presets
async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    presetsData = await res.json();
    const container = document.getElementById("preset-chips");
    container.innerHTML = "";

    presetsData.forEach((preset, index) => {
      const chip = document.createElement("button");
      chip.className = "preset-chip";
      chip.innerHTML = `${getIconForTag(preset.tag)}<span>${preset.category}</span>`;
      chip.onclick = () => selectPreset(index);
      container.appendChild(chip);
    });
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function getIconForTag(tag) {
  switch (tag) {
    case "easy":
      return `<svg class="chip-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
    case "high_risk":
      return `<svg class="chip-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#ff3434" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
    case "sensitive":
      return `<svg class="chip-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>`;
    case "safety_hazard":
      return `<svg class="chip-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>`;
    case "complaint":
      return `<svg class="chip-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#f97316" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="15" x2="12.01" y2="15"/></svg>`;
    default:
      return `<svg class="chip-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
  }
}

function selectPreset(index) {
  const preset = presetsData[index];
  if (!preset) return;

  const input = document.getElementById("query-input");
  input.value = preset.query;
  document.getElementById("char-count").textContent = `${preset.query.length} / 280 chars`;
  handleTriage();
}

// Execute Triage via POST /api/triage
async function handleTriage() {
  const input = document.getElementById("query-input");
  const query = input.value.trim();
  if (!query) {
    alert("Please enter a customer tweet to triage.");
    return;
  }

  const btn = document.getElementById("triage-btn");
  btn.disabled = true;
  btn.innerHTML = `<span>Triaging...</span>`;
  const t0 = performance.now();

  try {
    const res = await fetch("/api/triage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });

    if (!res.ok) {
      throw new Error(`Server returned status ${res.status}`);
    }

    const latencyMs = Math.round(performance.now() - t0);
    const data = await res.json();
    renderTriageResults(data, latencyMs);
  } catch (err) {
    console.error("Triage failed:", err);
    alert("Error communicating with triage API. Check console.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>Run Agent Triage</span><span class="btn-arrow">→</span>`;
  }
}

// Render Results into UI
function renderTriageResults(data, latencyMs = 28) {
  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("results-grid").classList.remove("hidden");

  // Update Latency Badge
  const latencyBadge = document.getElementById("latency-badge");
  if (latencyBadge) {
    latencyBadge.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg><span>${latencyMs}ms CPU</span>`;
  }

  // 1. Intent Tile
  const intentName = document.getElementById("intent-name");
  const intentConf = document.getElementById("intent-conf-badge");
  const progress = document.getElementById("intent-progress");

  intentName.textContent = data.predicted_intent.toUpperCase();
  const pct = (data.intent_confidence * 100).toFixed(1);
  intentConf.textContent = `${pct}% Conf`;
  progress.style.width = `${pct}%`;

  // 2. Escalation Tile
  const actionBadge = document.getElementById("escalation-action-badge");
  const reasonCode = document.getElementById("escalation-reason-code");
  const details = document.getElementById("escalation-details");

  actionBadge.textContent = data.escalation.decision === "AUTO_HANDLE" ? "AUTO-HANDLE" : "ESCALATE TO HUMAN";
  actionBadge.className = "escalation-action-badge " + (data.escalation.decision === "AUTO_HANDLE" ? "auto-handle" : "escalate");
  reasonCode.textContent = data.escalation.reason_code;
  details.textContent = data.escalation.reason_details;

  // 3. Retrieval Precedent
  const simBadge = document.getElementById("retrieval-sim-badge");
  const queryBox = document.getElementById("evidence-query");
  const replyBox = document.getElementById("evidence-reply");

  simBadge.textContent = `Cosine Sim: ${data.retrieval_similarity.toFixed(3)}`;

  if (data.retrieved_evidence && data.retrieved_evidence.length > 0) {
    const top = data.retrieved_evidence[0];
    queryBox.textContent = `"${top.historical_customer_query}"`;
    replyBox.textContent = `"${top.historical_brand_reply}"`;
  } else {
    queryBox.textContent = "No close historical match found in corpus.";
    replyBox.textContent = "No precedent available.";
  }

  // 4. Draft Reply
  const draftBubble = document.getElementById("reply-bubble");
  const groundedRating = document.getElementById("grounded-rating");

  draftBubble.textContent = `"${data.draft_reply}"`;
  groundedRating.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="#f59e0b" stroke="#f59e0b" stroke-width="1"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg><span>${data.groundedness_score.toFixed(1)} / 5.0 Grounded</span>`;
}

// Copy Reply to Clipboard with smooth animation
function copyReply() {
  const btn = document.getElementById("copy-reply-btn");
  const bubble = document.getElementById("reply-bubble");
  if (!bubble) return;
  const text = bubble.textContent.replace(/^[\s"]+|[\s"]+$/g, "");

  navigator.clipboard.writeText(text).then(() => {
    if (btn) {
      const origHTML = btn.innerHTML;
      btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg><span>Copied!</span>`;
      btn.classList.add("copied");
      setTimeout(() => {
        btn.innerHTML = origHTML;
        btn.classList.remove("copied");
      }, 2000);
    }
  }).catch(err => {
    console.error("Clipboard copy failed:", err);
  });
}

// Load Benchmark Metrics
async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();

    const tbody = document.getElementById("benchmark-table-body");
    tbody.innerHTML = "";

    if (data.baseline_comparison) {
      data.baseline_comparison.forEach(item => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>${item.System}</strong></td>
          <td>${item["Intent Accuracy"]}</td>
          <td>${item["Intent Macro F1"]}</td>
          <td>${item["Auto-Handle Coverage"]}</td>
          <td>${item["False Auto-Handle Rate (FAHR)"]}</td>
          <td>${item["Escalation Recall"]}</td>
          <td>${item["Groundedness (1-5)"]}</td>
        `;
        tbody.appendChild(row);
      });
    }
  } catch (err) {
    console.error("Failed to load metrics:", err);
  }
}

// Accordion Toggle
function toggleAccordion(button) {
  const item = button.parentElement;
  item.classList.toggle("open");
}
