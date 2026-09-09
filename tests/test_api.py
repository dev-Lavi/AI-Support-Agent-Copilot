"""Test suite for FastAPI REST endpoints in main.py."""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["brand"] == "@AppleSupport"


def test_root_browser_serves_html():
    response = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 200
    assert "AI Support Agent Copilot" in response.text
    assert "text/html" in response.headers.get("content-type", "")


def test_presets_endpoint():
    response = client.get("/api/presets")
    assert response.status_code == 200
    presets = response.json()
    assert len(presets) >= 4
    for p in presets:
        assert "query" in p
        assert "expected_intent" in p


def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "baseline_comparison" in data
    assert "judge_calibration" in data


def test_triage_endpoint():
    payload = {"query": "My iPhone 14 battery dies in 2 hours please help"}
    response = client.post("/api/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_intent"] == "hardware_battery_power"
    assert "escalation" in data
    assert data["escalation"]["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert "draft_reply" in data
