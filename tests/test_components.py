"""Comprehensive unit and integration test suite for the AI Support Agent."""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.cleaner import clean_tweet_text, is_valid_tweet
from src.data.splitter import create_grouped_splits
from src.intents.taxonomy import INTENTS, SENSITIVE_INTENTS
from src.intents.majority_baseline import MajorityClassBaseline
from src.intents.tfidf_baseline import TfidfBaselineClassifier
from src.escalation.policy import EscalationPolicy
from src.generation.grounded_drafter import GroundedReplyDrafter


def test_tweet_cleaner():
    raw = "Hey @AppleSupport! Check http://t.co/xyz123 &amp; @user2 ????"
    cleaned = clean_tweet_text(raw, preserve_brand_mention="AppleSupport")
    assert "<BRAND>" in cleaned
    assert "<URL>" in cleaned
    assert "<USER>" in cleaned
    assert "&" in cleaned
    assert "&amp;" not in cleaned
    assert is_valid_tweet(cleaned)


def test_split_isolation():
    dummy_df = pd.DataFrame({
        "conversation_id": [f"c_{i // 2}" for i in range(100)],
        "customer_text": [f"Query text {i}" for i in range(100)],
        "intent": ["hardware_battery_power"] * 100
    })
    train_df, val_df, test_df = create_grouped_splits(dummy_df, group_col="conversation_id")
    train_convs = set(train_df["conversation_id"])
    val_convs = set(val_df["conversation_id"])
    test_convs = set(test_df["conversation_id"])

    assert len(train_convs.intersection(val_convs)) == 0
    assert len(train_convs.intersection(test_convs)) == 0
    assert len(val_convs.intersection(test_convs)) == 0


def test_majority_baseline():
    b = MajorityClassBaseline()
    b.fit(["q1", "q2", "q3"], ["billing_subscription", "hardware_battery_power", "hardware_battery_power"])
    preds = b.predict(["unknown query"])
    assert preds[0] == "hardware_battery_power"
    probs = b.predict_proba(["unknown query"])
    assert np.max(probs) == 1.0


def test_tfidf_baseline():
    texts = [
        "iPhone battery drains fast",
        "iPhone battery dying in an hour",
        "How to cancel my subscription",
        "Cancel annual subscription bill"
    ]
    labels = [
        "hardware_battery_power",
        "hardware_battery_power",
        "billing_subscription",
        "billing_subscription"
    ]
    clf = TfidfBaselineClassifier()
    clf.fit(texts, labels)
    preds = clf.predict(["My battery is drained"])
    assert preds[0] == "hardware_battery_power"


def test_escalation_policy():
    policy = EscalationPolicy(tau_intent=0.70, tau_retrieval=0.65)

    # 1. Sensitive intent must escalate
    dec1 = policy.evaluate("My account is locked", "account_access_auth", 0.95, 0.90)
    assert dec1.action == "ESCALATE"
    assert dec1.reason_code == "SENSITIVE_CASE"

    # 2. Urgent keyword must escalate
    dec2 = policy.evaluate("I will file a lawsuit", "hardware_battery_power", 0.95, 0.90)
    assert dec2.action == "ESCALATE"
    assert dec2.reason_code == "HIGH_RISK_KEYWORD_DETECTED"

    # 3. Low confidence must escalate
    dec3 = policy.evaluate("Phone issue", "hardware_battery_power", 0.40, 0.90)
    assert dec3.action == "ESCALATE"
    assert dec3.reason_code == "LOW_INTENT_CONFIDENCE"

    # 4. Low retrieval similarity must escalate
    dec4 = policy.evaluate("Phone issue", "hardware_battery_power", 0.95, 0.40)
    assert dec4.action == "ESCALATE"
    assert dec4.reason_code == "INSUFFICIENT_HISTORICAL_EVIDENCE"

    # 5. Safe FAQ should auto-handle
    dec5 = policy.evaluate("Battery drains quickly", "hardware_battery_power", 0.92, 0.85)
    assert dec5.action == "AUTO_HANDLE"
    assert dec5.reason_code == "HIGH_CONFIDENCE_GROUNDED"


def test_grounded_drafter():
    drafter = GroundedReplyDrafter()
    evidence = [({
        "pair_id": "pair_001",
        "brand_text": "Check Battery Health under Settings > Battery: <URL> ^AB",
        "customer_text": "Battery drains fast"
    }, 0.85)]
    draft = drafter.draft_reply("Battery drains quickly", "hardware_battery_power", evidence)
    assert "Settings > Battery" in draft["draft_reply"]
    assert draft["groundedness_score"] >= 4.0
    assert len(draft["draft_reply"]) <= 280
