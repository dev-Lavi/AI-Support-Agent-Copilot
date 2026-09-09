# 04 — Intent Taxonomy Specification

## 1. Principles of Taxonomy Design

A customer support intent taxonomy must balance **granularity** (actionability of routing) against **separability** (statistical distinction between classes). In customer service NLP, overly granular taxonomies (e.g. 50+ classes) degrade into semantic overlap and high inter-annotator disagreement. Conversely, overly coarse taxonomies (e.g. 3 classes) fail to inform operational resolution.

We target an optimal operational window of **8–10 intents** derived from empirical clusters in the selected brand's support corpus (illustrated below for Tech / Device Support, e.g., `@AppleSupport`):

---

## 2. Intent Definitions & Boundaries

### 1. `account_access_auth`
* **Definition**: Inquiries regarding Apple ID, password resets, two-factor authentication (2FA), account lockout, or iCloud login failures.
* **Positive Examples**:
  * *"Locked out of my Apple ID and cannot get the verification code on my trusted number."*
  * *"Forgot my iCloud password and the recovery email is no longer accessible."*
* **Negative Examples**:
  * *"Why was I charged $9.99 on my Apple ID?"* (Belongs to `billing_subscription`)
* **Boundary with Similar Intents**: Distinct from `billing_subscription` in that no payment or charge dispute is mentioned; focus is strictly identity verification and access credentials.
* **Escalation Predisposition**: **HIGH / MANDATORY ESCALATION**. Automated agents must never attempt credential bypass or give speculative recovery steps on account security.

### 2. `billing_subscription`
* **Definition**: Questions regarding unauthorized charges, recurring subscription cancellations, App Store receipts, refund requests, or payment method updates.
* **Positive Examples**:
  * *"Got charged for an app subscription I cancelled three weeks ago. How do I get a refund?"*
  * *"Need to update my expired debit card on my account."*
* **Negative Examples**:
  * *"My app crashes when I open the payment page."* (Belongs to `app_software_issue`)
* **Boundary with Similar Intents**: Must involve currency, transaction history, or paid subscription status.
* **Escalation Predisposition**: **CONDITIONAL**. Simple policy explanations ("How to view purchase history") can be auto-handled; active refund claims and dispute investigations escalate.

### 3. `hardware_battery_power`
* **Definition**: Physical device issues related to battery drain, overheating, failure to charge, damaged charging ports, or power cycling.
* **Positive Examples**:
  * *"My iPhone battery drops from 80% to dead in 30 minutes after updating."*
  * *"iPad won't turn on or show the charging icon even when plugged into wall power."*
* **Negative Examples**:
  * *"The screen is unresponsive only in Safari."* (Belongs to `app_software_issue`)
* **Boundary with Similar Intents**: Involves physical energy, battery health metric, thermal issues, or power failure.
* **Escalation Predisposition**: **AUTO-HANDLE ELIGIBLE** (Standard diagnostic steps: battery health check, reset procedure) unless physical damage / replacement is requested.

### 4. `os_system_update`
* **Definition**: Difficulties during or after operating system updates (iOS/macOS), storage space errors during download, boot loops, or bricked devices post-update.
* **Positive Examples**:
  * *"iOS update is stuck on 'Verifying update' for 4 hours."*
  * *"Not enough storage to install the new macOS patch even though I have 20GB free."*
* **Negative Examples**:
  * *"My battery dies quickly on the new iOS."* (Belongs to `hardware_battery_power`)
* **Boundary with Similar Intents**: Centers on the installation, verification, or boot behavior of the OS upgrade itself.
* **Escalation Predisposition**: **AUTO-HANDLE ELIGIBLE** (Standard recovery mode instructions, iTunes/Finder restore guide).

### 5. `connectivity_network`
* **Definition**: Issues connecting to Wi-Fi networks, cellular data drops, Bluetooth pairing failures, or AirDrop malfunction.
* **Positive Examples**:
  * *"Bluetooth refuses to pair with my car audio system since yesterday."*
  * *"iPhone keeps dropping Wi-Fi connection every few minutes."*
* **Negative Examples**:
  * *"Can't log into iCloud over Wi-Fi."* (Belongs to `account_access_auth` if credential related)
* **Boundary with Similar Intents**: The root failure is the network transport or RF protocol (Wi-Fi, Bluetooth, 4G/5G).
* **Escalation Predisposition**: **AUTO-HANDLE ELIGIBLE** (Reset Network Settings, toggle Airplane mode, forget network steps).

### 6. `app_software_issue`
* **Definition**: Third-party or first-party application crashes, freezing, display rendering glitches, or unexpected app terminations.
* **Positive Examples**:
  * *"Camera app screen goes completely black whenever I try to take a photo."*
  * *"Messages app crashes immediately upon opening a conversation."*
* **Negative Examples**:
  * *"Bluetooth doesn't work in Spotify."* (Belongs to `connectivity_network`)
* **Boundary with Similar Intents**: Isolated to specific application execution rather than system-wide power or OS boot failure.
* **Escalation Predisposition**: **AUTO-HANDLE ELIGIBLE** (Force quit, cache clear, app reinstall, OS compatibility check).

### 7. `repair_service_warranty`
* **Definition**: Inquiries about AppleCare+ coverage, repair booking, Genius Bar appointments, repair status tracking, or repair cost estimates.
* **Positive Examples**:
  * *"How much does a screen replacement cost for an iPhone 12 out of warranty?"*
  * *"Need to book an appointment at the nearest store to fix a cracked back glass."*
* **Negative Examples**:
  * *"Why won't my screen turn on?"* (Belongs to `hardware_battery_power`)
* **Boundary with Similar Intents**: Explicitly requests physical servicing, warranty validation, or appointment scheduling.
* **Escalation Predisposition**: **AUTO-HANDLE ELIGIBLE** for appointment links and warranty check URLs; escalate if customer demands repair fee waiver.

### 8. `feedback_complaint`
* **Definition**: Customer expressing anger, dissatisfaction with customer service, brand policy criticism, or threats to switch competitors without asking a technical question.
* **Positive Examples**:
  * *"Terrible customer service at your store today, completely rude staff!"*
  * *"Never buying another device from you again. Overpriced junk."*
* **Negative Examples**:
  * *"I hate this update, it broke my Bluetooth."* (Belongs to `connectivity_network` as an actionable technical fault)
* **Boundary with Similar Intents**: Emotional venting without a specific solvable technical request.
* **Escalation Predisposition**: **MANDATORY ESCALATION**. Human empathy and brand reputation protection required.

### 9. `other_unknown`
* **Definition**: Messages that are ambiguous, gibberish, out-of-domain (e.g., spam, memes), or lacking sufficient context to classify into the 8 operational intents.
* **Positive Examples**:
  * *"Help please!!!"*
  * *"Check out this crypto link."*
* **Justification**: An explicit catch-all class prevents the classifier from forcing ambiguous noise into high-confidence operational buckets.
* **Escalation Predisposition**: **MANDATORY ESCALATION** (`UNKNOWN_INTENT` or `AMBIGUOUS_QUERY`).

---

## 3. Expected Dataset Distribution

Based on preliminary Twitter customer service profiles, intent distribution is inherently skewed:
* `hardware_battery_power`: ~20%
* `app_software_issue`: ~18%
* `account_access_auth`: ~15%
* `os_system_update`: ~14%
* `billing_subscription`: ~12%
* `connectivity_network`: ~10%
* `repair_service_warranty`: ~6%
* `feedback_complaint`: ~3%
* `other_unknown`: ~2%

*Note: Macro F1 is our primary metric precisely because minority intents (`feedback_complaint`, `repair_service_warranty`) must not be obscured by high-frequency technical troubleshooting.*
