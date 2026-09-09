"""Formal Intent Taxonomy specification and mapping utilities for @AppleSupport."""

from typing import Dict, List, Set

# The 9-intent empirical taxonomy
INTENTS: List[str] = [
    "hardware_battery_power",
    "app_software_issue",
    "account_access_auth",
    "os_system_update",
    "billing_subscription",
    "connectivity_network",
    "repair_service_warranty",
    "feedback_complaint",
    "other_unknown",
]

# Sensitive or policy-restricted intents that mandate escalation
SENSITIVE_INTENTS: Set[str] = {
    "account_access_auth",  # Password reset, 2FA, Apple ID lockout
    "feedback_complaint",   # Customer anger, legal threats, complaints
    "other_unknown",        # Out-of-domain or unclassifiable
}

# Intents eligible for automated handling if grounded evidence is strong
AUTO_HANDLE_ELIGIBLE_INTENTS: Set[str] = {
    "hardware_battery_power",
    "app_software_issue",
    "os_system_update",
    "billing_subscription",
    "connectivity_network",
    "repair_service_warranty",
}

# Mapping label -> integer index and index -> label
INTENT2ID: Dict[str, int] = {intent: idx for idx, intent in enumerate(INTENTS)}
ID2INTENT: Dict[int, str] = {idx: intent for idx, intent in enumerate(INTENTS)}

# Formal human-readable descriptions for each intent
INTENT_DESCRIPTIONS: Dict[str, str] = {
    "hardware_battery_power": "Physical device power, rapid battery drain, charging failure, overheating, or failure to turn on.",
    "app_software_issue": "First-party or third-party app crashes, freezing, display rendering glitches, or unexpected terminations.",
    "account_access_auth": "Apple ID, iCloud credentials, password resets, two-factor authentication, or account security lockouts.",
    "os_system_update": "Difficulties during or post OS updates (iOS/macOS), storage errors, stuck verification, or boot loops.",
    "billing_subscription": "Unauthorized charges, App Store receipts, subscription cancellations, refunds, or payment method updates.",
    "connectivity_network": "Wi-Fi dropouts, cellular data errors, Bluetooth pairing failures, or AirDrop transfer issues.",
    "repair_service_warranty": "Genius Bar appointments, repair cost estimates, AppleCare coverage, or hardware replacement inquiries.",
    "feedback_complaint": "Emotional complaints, dissatisfaction with staff or policy, or threats to switch without actionable technical requests.",
    "other_unknown": "Ambiguous, gibberish, out-of-domain, or unclassifiable inquiries lacking actionable context.",
}
