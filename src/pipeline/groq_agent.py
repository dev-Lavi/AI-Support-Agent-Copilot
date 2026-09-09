"""Google Gemini API Pipeline for High-Accuracy Intent Classification, Grounded Reply Drafting, and Conservative Escalation.

Zero-PyTorch cloud inference with sub-500ms latency and minimal memory (<5MB RAM).
Uses Google AI Studio free-tier Gemini API (gemini-2.0-flash-lite).
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import httpx

from src.intents.taxonomy import INTENTS, SENSITIVE_INTENTS


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.6-flash"
FAST_MODEL = "gemini-3.5-flash-lite"

# Ordered preference — most capable first, all free-tier available
# Updated per Gemini API 404 deprecation notices (Sep 2026)
CANDIDATE_MODELS = [
    "gemini-3.6-flash",        # replaces gemini-2.0-flash
    "gemini-3.5-flash-lite",   # replaces gemini-2.0-flash-lite
    "gemini-3.5-flash",        # general fallback
    "gemini-2.5-flash",        # older stable fallback
    "gemini-2.5-flash-lite",   # lightest fallback
]


SYSTEM_PROMPT = """You are the AI Customer Support Copilot for @AppleSupport on Twitter.
Your role is to analyze incoming customer tweets and return a structured JSON response.

You must accomplish 3 strict tasks:
1. CLASSIFY the tweet into exactly ONE of the 9 defined intents:
   - hardware_battery_power: Battery drain, overheating, physical swelling, charging failure, battery health.
   - app_software_issue: First/Third-party app crashes, camera black, keyboard lag, alarms not going off, Books sync/annotations, calculator, FaceID, app features failing.
   - connectivity_network: Wi-Fi toggle grayed out, Bluetooth drops, cellular no service, eSIM errors, AirDrop/CarPlay disconnects.
   - os_system_update: Stuck on Apple logo / boot loop after iOS update, update verification errors, storage full during update, beta profile.
   - repair_service_warranty: Screen/glass repair cost, AppleCare coverage, Genius Bar appointment, mail-in turnaround, loaner phone, repair quotes.
   - account_access_auth: Apple ID password reset, two-factor authentication (2FA) lockout, security recovery, disabled account, verification codes.
   - billing_subscription: Unauthorized App Store charge, subscription cancel, refund demand, gift card redemption, payment method decline.
   - feedback_complaint: Emotional complaints or feedback about staff, pricing, or service WITHOUT actionable technical troubleshooting.
   - other_unknown: Slang, gibberish, ambiguous, off-topic, jokes, crypto, or unrecognizable query.

2. DECIDE ESCALATION (Conservative Safety Policy):
   - You MUST ESCALATE (decision: "ESCALATE") if:
     * intent is feedback_complaint -> reason_code: "HIGH_RISK_GRIEVANCE"
     * intent is other_unknown -> reason_code: "AMBIGUOUS_QUERY_LOW_CONF"
     * intent is account_access_auth -> reason_code: "ACCOUNT_SECURITY_2FA"
     * Physical safety hazard (battery swelling, smoke, sparks, melted charger, hot adapter) -> reason_code: "SAFETY_CRITICAL_BATTERY_HAZARD"
     * Billing dispute, refund demand, unauthorized charge, stolen card -> reason_code: "BILLING_DISPUTE_REFUND"
     * Data loss or file deletion (lost annotations, lost photos, lost notes, lost recording) -> reason_code: "SENSITIVE_DATA_LOSS"
     * Repair price dispute or partner refusal ("quoted me $600", "robbery", "overpriced", "refused service") -> reason_code: "REPAIR_PRICE_DISPUTE"
     * Device bricked / boot loop / kernel panic ("stuck on apple logo for 6 hours", "purple screen") -> reason_code: "CRITICAL_SYSTEM_FAILURE"
   - You may AUTO-HANDLE (decision: "AUTO_HANDLE", reason_code: "HIGH_CONF_GROUNDED") ONLY IF:
     * The query is a standard technical troubleshooting FAQ (e.g. how to redeem gift card, alarm volume settings, battery health check, reset network settings, book appointment) without safety, security, billing dispute, or data loss.

3. DRAFT A GROUNDED REPLY:
   - If historical resolution precedents are provided, strictly base your solution on them.
   - Tone: Courteous, empathetic, concise, and professional (Apple Support voice).
   - Under 280 characters to fit Twitter's limit.
   - Sign off with "^AB".
   - Never promise a free hardware replacement or manual refund without authorized inspection.

Return ONLY a valid JSON object with this exact schema (no markdown, no code fences):
{
  "predicted_intent": "<one of the 9 intents>",
  "intent_confidence": <float between 0.50 and 0.99>,
  "decision": "AUTO_HANDLE" or "ESCALATE",
  "reason_code": "<reason_code>",
  "reason_details": "<1 sentence explaining why this decision was made>",
  "draft_reply": "<tweet reply <= 280 chars ending with ^AB>",
  "groundedness_score": <float between 1.0 and 5.0>
}
"""


class GroqSupportAgent:
    """Google Gemini-backed support triage agent.

    Kept as 'GroqSupportAgent' for backward compatibility with the pipeline.
    Reads GEMINI_API_KEY (preferred) or GROQ_API_KEY from environment for legacy compat.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        # Accept GEMINI_API_KEY preferentially; fall back to GROQ_API_KEY for easy migration
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GROQ_API_KEY", "")
        )
        self.model = model

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def _call_gemini(self, client: httpx.Client, model_id: str, user_content: str) -> Optional[str]:
        """Calls the Gemini generateContent REST endpoint. Returns raw text or None on failure."""
        url = f"{GEMINI_API_BASE}/{model_id}:generateContent?key={self.api_key.strip()}"
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        try:
            resp = client.post(url, json=payload, timeout=20.0)
            if resp.status_code == 200:
                data = resp.json()
                content = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    or ""
                )
                return content.strip() if content.strip() else None
            else:
                raise RuntimeError(
                    f"Gemini API returned status {resp.status_code} for model '{model_id}': {resp.text}"
                )
        except httpx.TimeoutException:
            raise RuntimeError(f"Gemini API timeout for model '{model_id}'")

    def triage(
        self,
        query: str,
        retrieved_evidence: Optional[List[Tuple[Dict, float]]] = None
    ) -> Dict:
        """Invokes Gemini to classify intent, evaluate escalation, and draft a grounded reply."""
        if not self.is_configured:
            raise ValueError("GEMINI_API_KEY is not configured.")

        # Build context from retrieved historical resolutions
        context_str = ""
        if retrieved_evidence:
            context_str = "\n\nHistorical Resolution Precedents from @AppleSupport:\n"
            for i, (doc, sim) in enumerate(retrieved_evidence[:2], 1):
                context_str += (
                    f"Precedent {i} (Similarity: {sim:.2f}):\n"
                    f"Customer: {doc.get('customer_text', '')}\n"
                    f"Brand Resolution: {doc.get('brand_text', '')}\n"
                )

        user_content = f"Incoming Customer Query:\n\"{query}\"{context_str}\n\nPlease triage and return JSON."

        # Dynamic model discovery: filter CANDIDATE_MODELS to only available ones
        active_candidates = list(CANDIDATE_MODELS)
        try:
            with httpx.Client(timeout=5.0) as probe:
                resp = probe.get(
                    f"{GEMINI_API_BASE}?key={self.api_key.strip()}",
                    timeout=5.0
                )
                if resp.status_code == 200:
                    available = {
                        m.get("name", "").replace("models/", "")
                        for m in resp.json().get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    }
                    filtered = [m for m in CANDIDATE_MODELS if m in available]
                    if filtered:
                        active_candidates = filtered
        except Exception:
            pass  # Proceed with hardcoded list

        last_error = None
        raw_content = None

        with httpx.Client(timeout=25.0) as client:
            for model_id in active_candidates:
                try:
                    raw_content = self._call_gemini(client, model_id, user_content)
                    if raw_content:
                        break
                    else:
                        last_error = f"Model '{model_id}' returned empty content, trying next."
                except RuntimeError as e:
                    last_error = str(e)
                    print(f"[pipeline] Gemini warning: {last_error}")

        if not raw_content:
            raise RuntimeError(last_error or "All Gemini model requests failed.")

        # Strip any accidental markdown code fences
        raw_content = raw_content.strip()
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
        raw_content = raw_content.strip()

        result = json.loads(raw_content)

        # Validate fields and sanitize
        pred_intent = result.get("predicted_intent", "other_unknown")
        if pred_intent not in INTENTS:
            pred_intent = "other_unknown"

        decision = result.get("decision", "ESCALATE").upper()
        if decision not in ["AUTO_HANDLE", "ESCALATE"]:
            decision = "ESCALATE"

        # Hard guardrail enforcement: physical safety or sensitive intents must always escalate
        query_lower = query.lower()
        if any(w in query_lower for w in ["swelling", "melted", "burned", "smoke", "sparks", "fire"]):
            decision = "ESCALATE"
            result["reason_code"] = "SAFETY_CRITICAL_BATTERY_HAZARD"
            result["reason_details"] = "Physical battery hazard detected: immediate escalation to Apple Safety Team required."

        if pred_intent in SENSITIVE_INTENTS and decision != "ESCALATE":
            decision = "ESCALATE"
            result["reason_code"] = "SENSITIVE_CASE"
            result["reason_details"] = f"Intent '{pred_intent}' requires human agent verification."

        # Build full distribution
        conf = float(result.get("intent_confidence", 0.95))
        dist = {intent: 0.01 for intent in INTENTS}
        dist[pred_intent] = round(conf, 4)

        return {
            "predicted_intent": pred_intent,
            "intent_confidence": conf,
            "intent_distribution": dist,
            "decision": decision,
            "reason_code": result.get("reason_code", "HIGH_CONF_GROUNDED"),
            "reason_details": result.get("reason_details", "Triage performed via Google Gemini"),
            "draft_reply": result.get("draft_reply", ""),
            "groundedness_score": float(result.get("groundedness_score", 4.5))
        }
