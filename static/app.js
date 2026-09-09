/**
 * Hiver AI Customer Support Copilot — Frontend Application Logic
 */

// State
let presetsData = [];

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  loadPresets();
  loadMetrics();
  setupTextareaCounter();
});

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
      counter.style.color = "var(--text-muted)";
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
      chip.textContent = `${getEmojiForTag(preset.tag)} ${preset.category}`;
      chip.onclick = () => selectPreset(index);
      container.appendChild(chip);
    });
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function getEmojiForTag(tag) {
  switch (tag) {
    case "easy": return "🟢";
    case "high_risk": return "🔴";
    case "sensitive": return "🟡";
    case "safety_hazard": return "🟣";
    case "complaint": return "🟠";
    default: return "🔹";
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
    latencyBadge.textContent = `⏱️ ${latencyMs}ms CPU`;
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
  groundedRating.textContent = `⭐ ${data.groundedness_score.toFixed(1)} / 5.0 Grounded`;
}

// Copy Reply to Clipboard with smooth animation
function copyReply() {
  const btn = document.getElementById("copy-reply-btn");
  const bubble = document.getElementById("reply-bubble");
  if (!bubble) return;
  const text = bubble.textContent.replace(/^[\s"]+|[\s"]+$/g, "");

  navigator.clipboard.writeText(text).then(() => {
    if (btn) {
      const origText = btn.textContent;
      btn.textContent = "✓ Copied!";
      btn.classList.add("copied");
      setTimeout(() => {
        btn.textContent = origText;
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
