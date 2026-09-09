"""FastAPI Backend Server for the AI Customer Support Agent.

Exposes REST API endpoints for Next.js / React frontend integration and Hugging Face Spaces deployment:
- POST /api/triage: Triages an incoming customer tweet (intent, retrieval, escalation, grounded draft)
- GET /api/metrics: Returns headline evaluation metrics, baseline comparisons, and judge agreement
- GET /api/presets: Returns curated test examples (easy FAQ, ambiguous, sensitive escalation)
"""

import os
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline.agent import SupportAgentPipeline

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(
    title="AI Customer Support Agent API (@AppleSupport)",
    description="Trustworthy, retrieval-grounded, conservatively escalating customer support copilot.",
    version="1.0.0"
)

# Enable CORS for Next.js / React / Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount results/plots for static image serving (confusion matrix heatmap)
plots_dir = PROJECT_ROOT / "results/plots"
if plots_dir.exists():
    app.mount("/results/plots", StaticFiles(directory=str(plots_dir)), name="plots")
_pipeline: Optional[SupportAgentPipeline] = None


def get_pipeline() -> SupportAgentPipeline:
    """Lazy loads or initializes the lightweight pipeline (<50MB RAM)."""
    global _pipeline
    if _pipeline is None:
        models_dir = PROJECT_ROOT / "results/models"
        _pipeline = SupportAgentPipeline.load(str(models_dir), prefer_lightweight=True)
    return _pipeline


class QueryRequest(BaseModel):
    query: str


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Silences browser 404 for favicon."""
    fav = PROJECT_ROOT / "static" / "favicon.ico"
    if fav.exists():
        return FileResponse(fav)
    return Response(status_code=204)


@app.get("/api/health")
@app.get("/api")
def health_check():
    """Health check and API index."""
    pipeline = get_pipeline()
    is_cloud = bool(pipeline.groq_agent and pipeline.groq_agent.is_configured)
    engine = "Google Gemini 2.0-Flash (Cloud RAG)" if is_cloud else "Scikit-Learn TF-IDF (Lightweight Fallback, <50MB RAM)"
    return {
        "status": "online",
        "brand": "@AppleSupport",
        "engine": engine,
        "assignment": "Hiver SDE Intern Take-Home",
        "docs_url": "/docs",
        "api_endpoints": {
            "triage": "POST /api/triage",
            "metrics": "GET /api/metrics",
            "presets": "GET /api/presets"
        }
    }


@app.get("/")
def root(request: Request):
    """Serves the interactive web UI dashboard for browsers, or API info for JSON clients."""
    accept = request.headers.get("accept", "")
    index_file = PROJECT_ROOT / "static" / "index.html"
    # If requested by a web browser, serve the interactive web UI dashboard
    if "text/html" in accept and index_file.exists():
        return FileResponse(index_file)
    # Default fallback / programmatic JSON status
    return health_check()


@app.post("/api/triage")
def triage_query(req: QueryRequest):
    """Triages an incoming customer message."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    pipeline = get_pipeline()
    res = pipeline.process(req.query)
    output = res.to_dict()
    gc.collect()
    return output


@app.get("/api/metrics")
def get_metrics():
    """Returns baseline comparisons and headline evaluation summary."""
    metrics_dir = PROJECT_ROOT / "results/metrics"
    comp_file = metrics_dir / "baseline_comparison.json"
    judge_file = metrics_dir / "judge_calibration.json"
    esc_file = metrics_dir / "escalation_metrics.json"

    comparison = []
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            comparison = json.load(f)

    judge_calibration = {}
    if judge_file.exists():
        with open(judge_file, "r", encoding="utf-8") as f:
            judge_calibration = json.load(f)

    escalation = {}
    if esc_file.exists():
        with open(esc_file, "r", encoding="utf-8") as f:
            escalation = json.load(f)

    return {
        "baseline_comparison": comparison,
        "judge_calibration": judge_calibration,
        "escalation_summary": escalation.get("final_agent", {})
    }


@app.get("/api/confusion-matrix")
def get_confusion_matrix():
    """Serves the normalized confusion matrix PNG image."""
    img_path = PROJECT_ROOT / "results" / "plots" / "confusion_matrix.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix image not found. Run scripts/evaluate.py first.")
    return FileResponse(str(img_path), media_type="image/png")


@app.get("/api/presets")
def get_presets():
    """Returns representative test cases from the Golden Set across all 9 intents for UI testing."""
    return [
        {
            "category": "Battery Drain FAQ",
            "query": "My iPhone 13 battery drains from 100% to 20% in 3 hours while idle. Battery health says 79%.",
            "expected_intent": "hardware_battery_power",
            "expected_decision": "AUTO_HANDLE",
            "tag": "easy"
        },
        {
            "category": "App Crash Glitch",
            "query": "Instagram crashes every single time I try to upload a story on iOS 17.1.",
            "expected_intent": "app_software_issue",
            "expected_decision": "AUTO_HANDLE",
            "tag": "easy"
        },
        {
            "category": "WiFi / Bluetooth Drop",
            "query": "My iPhone 12 constantly disconnects from home WiFi and the toggle button is grayed out.",
            "expected_intent": "connectivity_network",
            "expected_decision": "AUTO_HANDLE",
            "tag": "easy"
        },
        {
            "category": "iOS Update Loop",
            "query": "Updated to iOS 17.2 last night and now my phone is stuck on the Apple logo with spinning wheel.",
            "expected_intent": "os_system_update",
            "expected_decision": "AUTO_HANDLE",
            "tag": "easy"
        },
        {
            "category": "Screen Repair & Warranty",
            "query": "How much does it cost to replace a cracked back glass on an iPhone 14 Pro if I do not have AppleCare+?",
            "expected_intent": "repair_service_warranty",
            "expected_decision": "AUTO_HANDLE",
            "tag": "easy"
        },
        {
            "category": "Urgent Billing Dispute",
            "query": "I was charged $89.99 for an annual subscription I never downloaded! Refund my money immediately!",
            "expected_intent": "billing_subscription",
            "expected_decision": "ESCALATE",
            "tag": "high_risk"
        },
        {
            "category": "2FA Security Lockout",
            "query": "Locked out of my Apple ID because my old phone number is gone and I cannot get the two factor code.",
            "expected_intent": "account_access_auth",
            "expected_decision": "ESCALATE",
            "tag": "sensitive"
        },
        {
            "category": "Physical Battery Hazard",
            "query": "HELP my iPhone is extremely hot to the touch and the battery is visibly swelling the screen up!!",
            "expected_intent": "hardware_battery_power",
            "expected_decision": "ESCALATE",
            "tag": "safety_hazard"
        },
        {
            "category": "Severe Customer Grievance",
            "query": "Worst customer service on the planet. Your manager in Austin was rude, dismissive, and lied to me.",
            "expected_intent": "feedback_complaint",
            "expected_decision": "ESCALATE",
            "tag": "complaint"
        },
        {
            "category": "Ambiguous / Slang Query",
            "query": "idk my jawn is glitching out big time fam what do i even do rn",
            "expected_intent": "other_unknown",
            "expected_decision": "ESCALATE",
            "tag": "sensitive"
        }
    ]


# Mount static frontend files for web UI dashboard
static_dir = PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
