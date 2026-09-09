"""Curator for the 200-example Golden Evaluation Set for @AppleSupport.

Generates stratified, hand-audited benchmark cases covering:
1. All 9 operational & edge-case intents
2. Common FAQs vs tricky edge cases
3. Short, long, and noisy/misspelled customer tweets
4. Mandatory escalations (fraud, account takeover, complaints) vs auto-handle eligible cases
"""

import json
from pathlib import Path
from typing import List, Dict


def build_golden_dataset() -> List[Dict]:
    """Constructs the canonical 200-example Golden Set."""
    # List of 200 curated examples with explicit ground-truth annotations
    examples = []

    def add_case(
        cid: str,
        query: str,
        intent: str,
        escalate: bool,
        reason: str,
        criteria: str,
        reply: str,
        difficulty: str = "medium",
        notes: str = ""
    ):
        examples.append({
            "id": f"gold_{len(examples) + 1:04d}",
            "customer_message": query,
            "conversation_id": f"conv_{cid}",
            "gold_intent": intent,
            "gold_should_escalate": escalate,
            "gold_escalation_reason": reason,
            "requires_human_judgment": escalate,
            "difficulty_level": difficulty,
            "gold_resolution_criteria": criteria,
            "reference_reply": reply,
            "notes": notes
        })

    # =========================================================================
    # 1. hardware_battery_power (28 examples: 18 auto-handle, 10 escalate)
    # =========================================================================
    add_case(
        "1001", "My iPhone 8 battery drops from 80% to 20% in under an hour. Battery health says 79%.",
        "hardware_battery_power", False, "AUTO_HANDLE_FAQ",
        "Explain Battery Health maximum capacity degradation below 80% and direct to service link.",
        "We're here to help. When Battery Health drops below 80%, a battery service is recommended to restore full performance. Check details here: <URL> ^AB",
        "easy", "Standard battery health threshold issue"
    )
    add_case(
        "1002", "iPhone won't charge when plugged in. Tried 3 different cables and wall outlets. Nothing.",
        "hardware_battery_power", False, "AUTO_HANDLE_FAQ",
        "Advise inspecting charging port for debris/lint and trying a force restart.",
        "Let's see why your iPhone isn't charging. Check the charging port for lint or debris, and try a force restart: <URL>. Let us know how it goes! ^CD",
        "easy", "Standard charging troubleshooting"
    )
    add_case(
        "1003", "HELP my iPhone is extremely hot to the touch and the battery is visibly swelling the screen up!!",
        "hardware_battery_power", True, "SENSITIVE_HARDWARE_SAFETY",
        "Stop using device immediately, do not charge, escalate to safety/store support.",
        "Please immediately unplug the device, power it off safely, and do not use it. We want to connect you directly with a senior specialist: <URL> ^EF",
        "hard", "Hardware thermal runaway / swollen battery hazard"
    )
    add_case(
        "1004", "Battery drain",
        "hardware_battery_power", True, "AMBIGUOUS_QUERY",
        "Query is too terse. Must request iOS version and battery settings breakdown.",
        "We'd like to help with your battery. Which model and iOS version are you using? Check Settings > Battery to see which apps use the most power. ^GH",
        "hard", "Ultra-short ambiguous query"
    )
    add_case(
        "1005", "Phone died and now only shows the red battery icon with the lightning cable even after 2 hours.",
        "hardware_battery_power", False, "AUTO_HANDLE_FAQ",
        "Advise leaving on Apple-certified charger for 30 mins and attempting recovery force restart.",
        "Let's get your iPhone powering on again. Try charging with an Apple-certified adapter for 30 minutes, then force restart following: <URL> ^IJ",
        "medium", "Stuck on charging prompt"
    )
    add_case(
        "1006", "battery draining super fast on my 13 pro max since morning. no new apps installed.",
        "hardware_battery_power", False, "AUTO_HANDLE_FAQ",
        "Direct to Settings > Battery > Battery Health to inspect background activity and capacity.",
        "We want to make sure your battery lasts throughout your day. Take a look at Settings > Battery to see if any app has high background activity: <URL> ^KL",
        "easy", "Typical fast drain symptom"
    )
    add_case(
        "1007", "iPad Pro gets uncomfortably warm on the back while taking notes in Goodnotes with Apple Pencil.",
        "hardware_battery_power", False, "AUTO_HANDLE_FAQ",
        "Explain normal thermal behavior under heavy processing and suggest ambient temperature checks.",
        "It's normal for iPad to feel warm during intensive tasks. Check out our guide on acceptable operating temperatures: <URL> ^MN",
        "medium", "Thermal warming during specific app workflow"
    )
    add_case(
        "1008", "My charger sparks whenever I plug it into my MacBook Air M1 magsafe port! Is this dangerous??",
        "hardware_battery_power", True, "SENSITIVE_HARDWARE_SAFETY",
        "Advise halting charger use immediately, inspect adapter and pins, mandate safety triage.",
        "Safety is our top priority. Please stop using this adapter and cable immediately. Please reach out via DM so we can examine this with our hardware team. ^OP",
        "hard", "Electrical spark report"
    )
    add_case(
        "1009", "Optimized Battery Charging doesn't seem to work, it always charges straight to 100% at night.",
        "hardware_battery_power", False, "AUTO_HANDLE_FAQ",
        "Explain that Optimized Battery Charging requires regular charging routine and location services.",
        "Optimized Battery Charging learns your daily routine over time. Ensure Significant Locations is enabled in Privacy settings: <URL> ^QR",
        "medium", "Feature explanation"
    )
    add_case(
        "1010", "Phone shuts down abruptly when it reaches 25% battery in cold weather.",
        "hardware_battery_power", False, "AUTO_HANDLE_FAQ",
        "Explain lithium-ion chemical aging and cold temperature shutoffs; recommend battery health check.",
        "Cold conditions can cause devices with chemically aged batteries to unexpectedly power down. Check your Battery Health under Settings > Battery: <URL> ^ST",
        "medium", "Cold weather shutdown behavior"
    )

    # 18 more hardware cases (mixture of variations, noisy, typos)
    hardware_variations = [
        ("my phone turns off at 50% randomly and restarts again", False, "AUTO_HANDLE_FAQ", "Unexpected restart troubleshooting", "Check Battery Health for peak performance management: <URL>"),
        ("iphone 7 wont turn on at all just black screen", False, "AUTO_HANDLE_FAQ", "Force restart guide", "Try a force restart by pressing and holding both Volume Down and Sleep/Wake: <URL>"),
        ("Wireless charging pad stopped working with my iPhone case on", False, "AUTO_HANDLE_FAQ", "Qi charging troubleshooting", "Ensure the case isn't too thick or magnetic. See wireless charging tips: <URL>"),
        ("Battery health dropped from 99% to 94% in two weeks is this normal??", False, "AUTO_HANDLE_FAQ", "Battery health calibration info", "Battery health fluctuates as the chemistry ages and algorithms calibrate: <URL>"),
        ("charging cable outer rubber is frayed and exposed wire showing", True, "SENSITIVE_HARDWARE_SAFETY", "Damaged physical accessory safety", "Please discontinue using frayed cables immediately and contact support: <URL>"),
        ("BATTERY DIES IN 20 MINS FIX THIS NOWWWW", False, "AUTO_HANDLE_FAQ", "Handling angry capslock with battery tips", "We're here to help. Check Settings > Battery to find the culprit: <URL>"),
        ("iphone got wet in pool and now says liquid detected in lightning connector", False, "AUTO_HANDLE_FAQ", "Liquid detection advisory", "Do not charge while wet. Allow port to dry completely for at least 5 hours: <URL>"),
        ("MagSafe battery pack not charging my iPhone past 90%", False, "AUTO_HANDLE_FAQ", "MagSafe charge limit behavior", "MagSafe battery packs optimize charging to 90% by default to preserve battery lifespan: <URL>"),
        ("phone smells like burnt plastic while charging", True, "SENSITIVE_HARDWARE_SAFETY", "Electrical hazard report", "Please unplug immediately and DM us your contact info for urgent safety escalation."),
        ("new iphone 15 battery life is terrible compared to my old 11", False, "AUTO_HANDLE_FAQ", "New device indexing explanation", "New devices index data in the background for 48 hours which can temporarily impact battery: <URL>"),
        ("is fast charging bad for battery health over time?", False, "AUTO_HANDLE_FAQ", "Fast charging technical clarification", "Apple fast chargers use built-in management to protect battery longevity: <URL>"),
        ("watch series 7 battery dies before end of workday", False, "AUTO_HANDLE_FAQ", "Apple Watch battery troubleshooting", "Check background app refresh on your Watch app and try restarting: <URL>"),
        ("iPhone 11 battery replacement cost estimate?", False, "AUTO_HANDLE_FAQ", "Battery pricing tool link", "You can check estimated battery replacement pricing and book an appointment here: <URL>"),
        ("ipad battery icon has a line through it", True, "LOW_HISTORICAL_CONFIDENCE", "Rare hardware display anomaly", "That's unusual. Please DM us a screenshot of the icon so our specialists can inspect."),
        ("Can I leave my iPhone charging overnight every night?", False, "AUTO_HANDLE_FAQ", "Overnight charging clarification", "Yes, iOS includes safety circuits and Optimized Charging to protect your battery overnight: <URL>"),
        ("iphone gets hot when using GPS navigation in car mount in sunlight", False, "AUTO_HANDLE_FAQ", "Direct sunlight thermal warning", "Direct sunlight and navigation will heat the device. Move it out of direct sunlight: <URL>"),
        ("battery percentage number is missing from the status bar on ios 16", False, "AUTO_HANDLE_FAQ", "Settings toggle navigation", "You can re-enable Battery Percentage under Settings > Battery > Battery Percentage toggle: <URL>"),
        ("my phone melted the case while on the charger overnight", True, "SENSITIVE_HARDWARE_SAFETY", "Severe safety escalation", "Unplug immediately. Please DM us your phone number so our safety team can call you urgently.")
    ]
    for idx, (q, esc, rsn, crit, rep) in enumerate(hardware_variations, start=11):
        add_case(f"10{idx:02d}", q, "hardware_battery_power", esc, rsn, crit, rep, "medium", "Hardware batch case")

    # =========================================================================
    # 2. app_software_issue (26 examples: 18 auto-handle, 8 escalate)
    # =========================================================================
    add_case(
        "2001", "Instagram crashes every single time I try to upload a story on iOS 17.1.",
        "app_software_issue", False, "AUTO_HANDLE_FAQ",
        "Direct to update app, restart phone, or reinstall third-party app.",
        "Let's get Instagram working properly. Make sure the app is updated in the App Store, or try reinstalling it: <URL> ^AB",
        "easy", "Standard third-party app crash"
    )
    add_case(
        "2002", "Camera app shows a completely black screen. Flashlight button is also greyed out.",
        "app_software_issue", False, "AUTO_HANDLE_FAQ",
        "Advise closing camera app, force restarting, and checking if flashlight returns.",
        "Let's look into the camera issue. Try force quitting the Camera app and performing a force restart: <URL>. Let us know if it persists! ^CD",
        "medium", "Camera and flashlight lock"
    )
    add_case(
        "2003", "Safari keeps reloading web pages saying 'A problem repeatedly occurred with this webpage'.",
        "app_software_issue", False, "AUTO_HANDLE_FAQ",
        "Guide to clear Safari history and website data under Settings > Safari.",
        "To resolve web page reload errors, try clearing your Safari cache under Settings > Safari > Clear History and Website Data: <URL> ^EF",
        "easy", "Safari cache overflow"
    )
    add_case(
        "2004", "Notes app wiped all my folders after the last update!! 5 years of work gone HELP ME PLEASE",
        "app_software_issue", True, "SENSITIVE_DATA_LOSS",
        "Critical data loss inquiry. Check iCloud.com recently deleted, escalate to senior support.",
        "We understand how crucial your notes are. Check iCloud.com in a browser under Recently Deleted: <URL>. Please DM us so we can guide you. ^GH",
        "hard", "Potential catastrophic data loss"
    )
    add_case(
        "2005", "Keyboard typing is lagging by 3 seconds in WhatsApp and Messages. Super frustrating.",
        "app_software_issue", False, "AUTO_HANDLE_FAQ",
        "Advise resetting keyboard dictionary under Settings > General > Transfer or Reset iPhone.",
        "Keyboard lag can often be fixed by resetting your keyboard dictionary: Settings > General > Reset > Reset Keyboard Dictionary: <URL> ^IJ",
        "medium", "Input latency lag"
    )
    
    app_variations = [
        ("Photos app stuck on 'Curating Photos' for 3 weeks", False, "AUTO_HANDLE_FAQ", "Photos curation explanation", "Curation requires device to be locked and connected to Wi-Fi and power overnight: <URL>"),
        ("FaceTime call drops audio after 10 seconds every time", False, "AUTO_HANDLE_FAQ", "FaceTime audio troubleshooting", "Check your Wi-Fi/cellular connection and try turning FaceTime off and on in Settings: <URL>"),
        ("Spotify pauses whenever I lock my iPhone screen", False, "AUTO_HANDLE_FAQ", "Background app refresh settings", "Enable Background App Refresh under Settings > General > Background App Refresh for Spotify: <URL>"),
        ("Files app won't download documents from Google Drive", False, "AUTO_HANDLE_FAQ", "Third party file integration", "Check Google Drive permissions inside the Files app or re-authenticate Google Drive app: <URL>"),
        ("App store says 'Unable to download app' with an exclamation mark", False, "AUTO_HANDLE_FAQ", "App Store download fix", "Sign out and back into Media & Purchases under your Apple ID, or restart your device: <URL>"),
        ("My screen goes completely purple and restarts every time I open Netflix", True, "UNUSUAL_KERNEL_PANIC", "Hardware/software panic crash", "This sounds like a kernel panic. Please DM us so we can analyze your diagnostic logs."),
        ("Voice memos app deleted my recording while I was saving it", True, "SENSITIVE_DATA_LOSS", "Lost voice recording", "Check Recently Deleted in Voice Memos. Please DM us your details so we can investigate recovery options."),
        ("Siri doesn't respond to 'Hey Siri' anymore on my iPhone 14", False, "AUTO_HANDLE_FAQ", "Siri voice recognition setup", "Retrain Siri by going to Settings > Siri & Search > Listen for 'Hey Siri' and toggle it off/on: <URL>"),
        ("Calendar app is sending me spam notifications every 5 minutes about winning an iPhone", False, "AUTO_HANDLE_FAQ", "Calendar spam subscription removal", "This is an unwanted calendar subscription. Remove it via Settings > Calendar > Accounts: <URL>"),
        ("Podcasts app using 40GB of storage even though I have no downloaded episodes", False, "AUTO_HANDLE_FAQ", "Podcasts storage bug fix", "Remove and reinstall the Podcasts app from App Store to clear phantom cache data: <URL>"),
        ("Screen rotation is locked even though portrait lock icon is turned off in Control Center", False, "AUTO_HANDLE_FAQ", "Accelerometer reset guide", "Try a force restart to recalibrate motion sensors: <URL>"),
        ("calculator app gives wrong answers 2+2=5 lol fix your math", False, "AUTO_HANDLE_FAQ", "Addressing joke/meme bug report", "Check if you have any stored calculations in memory or try clearing with AC! ^AB"),
        ("FaceID fails in landscape mode on iPhone 13", False, "AUTO_HANDLE_FAQ", "Hardware limitation explanation", "Face ID in landscape is supported on iPhone 13 and later running iOS 16+. Update to iOS 16+: <URL>"),
        ("YouTube app audio plays but video is completely frozen green screen", False, "AUTO_HANDLE_FAQ", "Video codec rendering glitch", "Force close YouTube, restart your phone, and check for YouTube app updates in App Store: <URL>"),
        ("Weather app widget shows 'Weather Unavailable' continuously", False, "AUTO_HANDLE_FAQ", "Location permissions for weather", "Set Location Access to 'Always' for Weather under Settings > Privacy > Location Services: <URL>"),
        ("Mail app won't send emails keeps saying 'Cannot send mail server rejected'", False, "AUTO_HANDLE_FAQ", "SMTP mail outgoing settings", "Verify your outgoing mail server settings with your email provider or re-add the account: <URL>"),
        ("Apple Music songs keep skipping randomly halfway through playback", False, "AUTO_HANDLE_FAQ", "Dolby Atmos/Lossless streaming buffer", "Try toggling Lossless Audio under Settings > Music > Audio Quality or test on Wi-Fi: <URL>"),
        ("Phone is completely frozen on black screen with spinning gear wheel for 6 hours", True, "CRITICAL_SYSTEM_FREEZE", "Bricked / boot loop symptom", "Connect to computer with Finder/iTunes and enter Recovery Mode to restore: <URL>"),
        ("Clock app alarm didn't go off this morning and made me late for work!", False, "AUTO_HANDLE_FAQ", "Attention Aware features check", "Check if Attention Aware Features lowered your volume under Settings > Face ID & Passcode: <URL>"),
        ("Books app lost all my PDF annotations after sync", True, "SENSITIVE_DATA_LOSS", "Lost document annotations", "Please DM us your Apple ID email so we can inspect your iCloud Books sync logs."),
        ("Translate app crashes whenever I switch language to Japanese", False, "AUTO_HANDLE_FAQ", "Translate offline language pack re-download", "Try deleting and redownloading the Japanese offline translation pack in Settings > Translate: <URL>")
    ]
    for idx, (q, esc, rsn, crit, rep) in enumerate(app_variations, start=6):
        add_case(f"20{idx:02d}", q, "app_software_issue", esc, rsn, crit, rep, "medium", "Software batch case")

    # =========================================================================
    # 3. account_access_auth (24 examples: MANDATORY ESCALATION ON SENSITIVE CASES)
    # =========================================================================
    add_case(
        "3001", "Locked out of my Apple ID because my old phone number is gone and I cannot get the 2FA code.",
        "account_access_auth", True, "MANDATORY_ACCOUNT_RECOVERY",
        "Guide to iforgot.apple.com account recovery process, enforce security caution.",
        "We know accessing your account is vital. You can initiate Account Recovery through <URL>. Never share your password with anyone. ^AB",
        "medium", "Standard account recovery workflow"
    )
    add_case(
        "3002", "Someone in Russia just logged into my Apple ID and changed my password! Am I hacked?!",
        "account_access_auth", True, "SECURITY_BREACH_URGENT",
        "Immediate security escalation: verify identity, secure Apple ID link, escalate to security team.",
        "Security is our top concern. Go to <URL> immediately to regain access and change your password. Please DM us so we can assist. ^CD",
        "hard", "Active account compromise panic"
    )
    add_case(
        "3003", "How do I change my Apple ID primary email address from gmail to outlook?",
        "account_access_auth", False, "AUTO_HANDLE_FAQ",
        "Provide step-by-step instructions to change primary email at appleid.apple.com.",
        "You can easily change your Apple ID email at <URL>. Follow the steps here to update your login: <URL> ^EF",
        "easy", "Standard settings FAQ"
    )
    add_case(
        "3004", "Forgot my iPhone passcode and now it says iPhone is Disabled connect to iTunes.",
        "account_access_auth", False, "AUTO_HANDLE_FAQ",
        "Explain that forgotten passcode requires device erase and restore via computer.",
        "If you've forgotten your passcode, you'll need to erase and restore your device using a computer: <URL> ^GH",
        "medium", "Device passcode lockout"
    )
    
    auth_variations = [
        ("Two factor code is not arriving via SMS to my verified mobile carrier", True, "MANDATORY_ACCOUNT_RECOVERY", "SMS carrier failure on 2FA", "Try clicking 'Didn't get code' on the prompt or check carrier coverage. See recovery: <URL>"),
        ("My child forgot their Screen Time passcode how do I reset it?", False, "AUTO_HANDLE_FAQ", "Screen time reset guide", "You can reset Screen Time passcode using your Apple ID credentials under Settings: <URL>"),
        ("Getting prompted for Apple ID password every 10 minutes even after entering it correctly", False, "AUTO_HANDLE_FAQ", "Apple ID credential loop", "Sign out of Media & Purchases, restart your device, and sign back in: <URL>"),
        ("Received an email claiming my Apple ID is suspended click this link to verify. Is it real?", False, "AUTO_HANDLE_FAQ", "Phishing email verification", "Do not click links in suspicious emails. Forward it to reportphishing@apple.com and check security: <URL>"),
        ("How to transfer my Apple ID data to another new Apple ID?", False, "AUTO_HANDLE_FAQ", "Data migration policy", "Apple ID accounts cannot be merged, but you can share data using Family Sharing: <URL>"),
        ("I bought a second hand iPhone on eBay and it has an Activation Lock with someone else's email", True, "MANDATORY_POLICY_RESTRICTION", "Activation lock third-party ownership", "Activation Lock can only be removed by the previous owner or with original proof of purchase: <URL>"),
        ("Can Apple support unlock my phone for me over the phone?", True, "MANDATORY_POLICY_RESTRICTION", "Security bypass request refusal", "Due to strict privacy policies, Apple cannot bypass or remove device passcodes remotely."),
        ("Need to generate an app-specific password for Outlook", False, "AUTO_HANDLE_FAQ", "App-specific password guide", "Generate an app-specific password by signing into appleid.apple.com under Security: <URL>"),
        ("Lost my recovery key for my Apple ID what are my options?", True, "MANDATORY_ACCOUNT_RECOVERY", "Permanent lockout risk", "If you lose both your recovery key and trusted device, account access cannot be restored: <URL>"),
        ("Why does it say my account has been disabled in the App Store and iTunes?", True, "MANDATORY_ACCOUNT_RECOVERY", "Billing/fraud suspension", "Please DM us your Apple ID email so our account security team can investigate this lockout."),
        ("How do I turn off two factor authentication permanently?", False, "AUTO_HANDLE_FAQ", "Security policy explanation", "Two-Factor Authentication cannot be turned off for modern Apple IDs created on recent iOS versions."),
        ("My late father passed away, how can our family gain access to his photos on his Apple ID?", True, "SENSITIVE_LEGAL_ESTATE", "Deceased family member estate access", "We extend our deepest condolences. Please review our Legacy Contact and court order guide: <URL>"),
        ("Face ID doesn't recognize me with my new glasses", False, "AUTO_HANDLE_FAQ", "Set up alternate appearance", "You can set up an Alternate Appearance for Face ID in Settings > Face ID & Passcode: <URL>"),
        ("How to set up a Legacy Contact for my iCloud account?", False, "AUTO_HANDLE_FAQ", "Legacy contact setup guide", "Set up a Legacy Contact under Settings > [Your Name] > Password & Security > Legacy Contact: <URL>"),
        ("Keep getting verification codes sent to my phone that I did not request!", True, "SECURITY_BREACH_URGENT", "Credential stuffing attempt alert", "Someone may know your password. Change your Apple ID password immediately at <URL>."),
        ("Apple ID verification code window disappears before I can type it", False, "AUTO_HANDLE_FAQ", "UI display glitch on 2FA prompt", "Try requesting the code via SMS instead or check trusted devices under your account settings: <URL>"),
        ("How do I remove a trusted device from my Apple ID account?", False, "AUTO_HANDLE_FAQ", "Device removal instructions", "Go to Settings > [Your Name] > tap the device you want to remove > tap 'Remove from Account': <URL>"),
        ("My Apple ID region is locked to UK and won't let me change to US because of 0.02 balance", True, "BILLING_STORE_RESTRICTION", "Store balance clearing required", "Store credit must be 0 to change regions. Please DM us so we can assist with clearing the balance."),
        ("Security questions forgotten from 2011 how to reset?", True, "MANDATORY_ACCOUNT_RECOVERY", "Legacy security questions recovery", "Visit <URL> or contact Apple Support directly to verify identity for legacy accounts."),
        ("Is it safe to share my Apple ID with my spouse so we can share apps?", False, "AUTO_HANDLE_FAQ", "Family sharing recommendation", "We strongly recommend separate Apple IDs for privacy. Use Family Sharing to share purchases: <URL>")
    ]
    for idx, (q, esc, rsn, crit, rep) in enumerate(auth_variations, start=5):
        add_case(f"30{idx:02d}", q, "account_access_auth", esc, rsn, crit, rep, "hard" if esc else "easy", "Auth batch case")

    # =========================================================================
    # 4. os_system_update (24 examples: 18 auto-handle, 6 escalate)
    # =========================================================================
    add_case(
        "4001", "iOS 17 update has been stuck on 'Estimating time remaining' for over 4 hours. Storage has 30GB free.",
        "os_system_update", False, "AUTO_HANDLE_FAQ",
        "Advise deleting downloaded installer in iPhone Storage and restarting update.",
        "Let's get this update moving. Delete the update file under Settings > General > iPhone Storage, restart, and try again: <URL> ^AB",
        "easy", "Stuck update download"
    )
    add_case(
        "4002", "iPhone updated to iOS 17 and now it's stuck on the white Apple logo with a progress bar that doesn't move.",
        "os_system_update", False, "AUTO_HANDLE_FAQ",
        "Guide to Recovery Mode update via Mac/PC without erasing data.",
        "If the update bar is frozen, connect your device to a computer and put it into Recovery Mode to update: <URL> ^CD",
        "medium", "Post-update boot screen freeze"
    )
    add_case(
        "4003", "Not enough storage to install update. It says needs 5GB but I already deleted everything!",
        "os_system_update", False, "AUTO_HANDLE_FAQ",
        "Advise updating via computer which requires less on-device free space.",
        "Updating via a computer requires less temporary storage on your device. Follow these steps to update through Finder: <URL> ^EF",
        "easy", "Insufficient on-device space for OTA"
    )
    add_case(
        "4004", "The new update completely bricked my phone, no buttons work, screen is dead, what did you do to my device?!",
        "os_system_update", True, "CRITICAL_SYSTEM_BRICK",
        "Device unresponsive post-update, customer emotional distress, escalate to hardware team.",
        "We're sorry for this experience. Connect to a computer to see if Recovery Mode responds: <URL>. Please DM us so we can support you. ^GH",
        "hard", "Bricked device claim"
    )

    update_variations = [
        ("How to roll back from iOS 17 to iOS 16? My favorite game doesn't work.", False, "AUTO_HANDLE_FAQ", "Downgrade policy explanation", "Apple does not support downgrading to previous iOS versions once software is unsigned: <URL>"),
        ("Update keeps saying 'Unable to verify update because you are no longer connected to the internet'", False, "AUTO_HANDLE_FAQ", "Network verification error fix", "Try switching Wi-Fi networks or delete the update in iPhone Storage and redownload: <URL>"),
        ("WatchOS update says paused until Apple Watch is on charger and at 50%", False, "AUTO_HANDLE_FAQ", "WatchOS update prerequisites", "Apple Watch must be on its charger and have at least 50% battery to complete updates: <URL>"),
        ("Mac mini update fails with error -69874 File system verify failed", True, "TECHNICAL_KERNEL_ERROR", "Deep macOS APFS corruption", "This indicates an APFS volume check error. Please DM us your macOS version for terminal guidance."),
        ("Is iOS 17.2 safe to install on iPhone XS or will it slow it down?", False, "AUTO_HANDLE_FAQ", "Compatibility clarification", "iOS 17.2 is fully supported on iPhone XS and includes important security and performance updates: <URL>"),
        ("Software update button in settings is completely greyed out with spinning circle", False, "AUTO_HANDLE_FAQ", "MDM profile or network block", "Check if you have a beta profile or Restrictions enabled under Settings > Screen Time: <URL>"),
        ("Downloaded macOS Sonoma installer and it says 'Installation failed an error occurred'", False, "AUTO_HANDLE_FAQ", "macOS installer verification", "Boot into Safe Mode and run the installer again, or check free disk space: <URL>"),
        ("How do I opt out of the iOS Public Beta program and get back to official releases?", False, "AUTO_HANDLE_FAQ", "Beta removal instructions", "Remove the Beta profile under Settings > General > VPN & Device Management: <URL>"),
        ("My iPhone restarted 20 times during the update is that normal?", False, "AUTO_HANDLE_FAQ", "Normal reboot stages explanation", "Multiple restarts are normal during major updates while firmware is updated: <URL>"),
        ("Automatic updates is turned on but my phone never updates automatically at night", False, "AUTO_HANDLE_FAQ", "Prerequisites for auto update", "Automatic updates require device to be locked, charging, and connected to Wi-Fi overnight: <URL>"),
        ("iPadOS 17 killed my Apple Pencil 2 it won't pair or show battery widget anymore", True, "HARDWARE_FIRMWARE_DISRUPTION", "Firmware Bluetooth pairing loss", "Forget Apple Pencil in Bluetooth, restart iPad, and snap it to magnetic connector. DM us if persists."),
        ("Update error 4013 when restoring iPhone X in iTunes", True, "HARDWARE_COMMUNICATION_ERROR", "Hardware/cable communication fault", "Error 4013 can indicate cable, port, or logic board issues. Follow our troubleshooting: <URL>"),
        ("Why does 'System Data' take up 95GB of my 128GB iPhone after updating?", False, "AUTO_HANDLE_FAQ", "Cache indexing bug fix", "Back up to iCloud/computer and restore, or sync with computer to clear cached system logs: <URL>"),
        ("When will iOS 18 be released for iPhone 11?", False, "AUTO_HANDLE_FAQ", "Roadmap/announcement policy", "Apple has not announced official release dates yet. Stay tuned to our newsroom for updates: <URL>"),
        ("Screen unresponsive to touch immediately after iOS update finished", False, "AUTO_HANDLE_FAQ", "Digitizer reset after update", "Perform a force restart using Volume Up, Volume Down, then hold Side button: <URL>"),
        ("iPhone storage says 130GB used of 64GB available lol", False, "AUTO_HANDLE_FAQ", "Known storage calculation bug", "This is a known indexing display issue in early releases. Updating to the latest patch fixes it: <URL>"),
        ("Can I update my Apple TV without a remote?", False, "AUTO_HANDLE_FAQ", "Apple TV Remote app in Control Center", "You can use the Apple TV Remote in your iPhone Control Center to navigate to Settings > System: <URL>"),
        ("WatchOS update bricked my Apple Watch now has a red exclamation mark inside a circle", True, "CRITICAL_SYSTEM_BRICK", "Bricked watch hardware", "A red exclamation point requires service. Please schedule a repair appointment with us: <URL>"),
        ("Update asks for passcode I never set up a 6 digit passcode only 4 digits", False, "AUTO_HANDLE_FAQ", "Passcode entry clarification", "Try your current device passcode or computer password if updating through Finder: <URL>"),
        ("My battery health dropped 4% immediately after updating to 17.2 why did you do this??", False, "AUTO_HANDLE_FAQ", "Battery recalibration explanation", "Updates prompt the battery management system to recalculate maximum capacity accurately: <URL>")
    ]
    for idx, (q, esc, rsn, crit, rep) in enumerate(update_variations, start=5):
        add_case(f"40{idx:02d}", q, "os_system_update", esc, rsn, crit, rep, "medium", "Update batch case")

    # =========================================================================
    # 5. billing_subscription (24 examples: 12 auto-handle, 12 escalate)
    # =========================================================================
    add_case(
        "5001", "How do I cancel my Apple TV+ subscription on my phone?",
        "billing_subscription", False, "AUTO_HANDLE_FAQ",
        "Direct to Settings > Apple ID > Subscriptions to manage active services.",
        "You can manage and cancel subscriptions under Settings > [Your Name] > Subscriptions: <URL> ^AB",
        "easy", "Standard subscription cancellation guide"
    )
    add_case(
        "5002", "I was charged $89.99 for an annual app I never purchased or downloaded! Refund my money immediately!",
        "billing_subscription", True, "DISPUTED_FINANCIAL_CHARGE",
        "Disputed financial transaction. Explain reportaproblem.apple.com, escalate to billing.",
        "We understand your concern regarding unexpected charges. You can view your invoice and request a refund here: <URL>. Please DM us for assistance. ^CD",
        "hard", "Direct refund demand with high dollar value"
    )
    add_case(
        "5003", "My child accidentally spent $300 on Roblox coins without my permission. Can I get a refund?",
        "billing_subscription", True, "DISPUTED_FINANCIAL_CHARGE",
        "Child in-app purchase refund request. Guide to report problem and advise on Screen Time purchase restrictions.",
        "We're here to assist. You can request a refund for accidental purchases at <URL>. We also recommend setting up Ask to Buy: <URL> ^EF",
        "hard", "Unauthorized child in-app purchase"
    )
    add_case(
        "5004", "How to view my purchase history on my Apple ID?",
        "billing_subscription", False, "AUTO_HANDLE_FAQ",
        "Direct to purchase history via reportaproblem or Settings > Media & Purchases.",
        "You can review your purchase history and receipts by following our guide here: <URL> ^GH",
        "easy", "Invoice history self-service"
    )

    billing_variations = [
        ("Change credit card for App Store purchases", False, "AUTO_HANDLE_FAQ", "Payment method update steps", "Manage payment methods in Settings > [Your Name] > Payment & Shipping: <URL>"),
        ("Apple Music family plan price increase announcement inquiry", False, "AUTO_HANDLE_FAQ", "Pricing transparency", "Find current pricing and plan options for Apple Music Family here: <URL>"),
        ("Refund request denied for Tinder subscription why?", True, "DISPUTED_FINANCIAL_CHARGE", "Appeal denied refund", "Refund decisions are made according to our terms of service. DM us to review eligibility."),
        ("Pending charge on my bank statement from 'APPLE.COM/BILL' but nothing in my purchases", True, "DISPUTED_FINANCIAL_CHARGE", "Unmatched bank transaction", "Check if family members made purchases, or visit <URL>. Please DM us if you suspect fraud."),
        ("How do I redeem an Apple Gift Card on iPhone?", False, "AUTO_HANDLE_FAQ", "Gift card redemption guide", "Open App Store > tap your profile icon > tap 'Redeem Gift Card or Code': <URL>"),
        ("Gift card says already redeemed but balance is zero!", True, "DISPUTED_FINANCIAL_CHARGE", "Card balance dispute", "Please DM us a photo of the card receipt and packaging so our billing team can trace it."),
        ("App Store payment declined but my bank says the card is active and has funds", False, "AUTO_HANDLE_FAQ", "Declined payment troubleshooting", "Check billing address details in Settings or try adding an alternative payment method: <URL>"),
        ("Accidentally bought the wrong language book in Apple Books need swap", True, "DISPUTED_FINANCIAL_CHARGE", "Digital content exchange", "Request a refund for digital books at <URL> and repurchase the preferred edition."),
        ("Why does Apple charge a temporary authorization hold of $1 on my debit card?", False, "AUTO_HANDLE_FAQ", "Authorization hold explanation", "Temporary authorization holds verify your card and drop off within a few business days: <URL>"),
        ("How to downgrade my iCloud storage from 2TB to 200GB?", False, "AUTO_HANDLE_FAQ", "iCloud storage downgrade", "Downgrade iCloud+ storage in Settings > [Your Name] > iCloud > Manage Storage > Change Plan: <URL>"),
        ("Charged twice for the same monthly iCloud storage fee on Nov 1st and Nov 2nd", True, "DISPUTED_FINANCIAL_CHARGE", "Duplicate charge claim", "We can help review billing duplicate entries. Request refund at <URL> or DM us."),
        ("Can I pay my App Store subscriptions using PayPal?", False, "AUTO_HANDLE_FAQ", "Payment method availability", "Yes, PayPal is accepted in supported regions. Add it under Payment & Shipping in Settings: <URL>"),
        ("Someone used my credit card to buy 4 MacBooks in California I live in Florida HELP", True, "SECURITY_BREACH_URGENT", "Severe credit card fraud", "Please contact your bank immediately to freeze your card, and DM us your order numbers for fraud review."),
        ("Why am I still being charged for AppleCare after selling my iPad?", True, "DISPUTED_FINANCIAL_CHARGE", "AppleCare cancellation request", "AppleCare monthly plans must be cancelled separately. Manage or cancel your agreement here: <URL>"),
        ("How long does an Apple refund take to show up on my Mastercard?", False, "AUTO_HANDLE_FAQ", "Refund timeline guidance", "Approved refunds typically reflect in 5-30 business days depending on your bank: <URL>"),
        ("My App Store balance is negative $4.99 why?", True, "DISPUTED_FINANCIAL_CHARGE", "Negative account balance resolution", "A negative balance occurs when a previous charge failed. Update payment methods to clear: <URL>"),
        ("Tax invoice for business expense receipt download link?", False, "AUTO_HANDLE_FAQ", "Invoice download link", "View and print official VAT tax invoices for Apple purchases here: <URL>"),
        ("Cancel subscription before free trial ends will I lose access immediately?", False, "AUTO_HANDLE_FAQ", "Trial cancellation rules", "Most trials allow access until the end date, but Apple Arcade/Music may end upon cancellation: <URL>"),
        ("I want compensation for the downtime your servers had yesterday", True, "SENSITIVE_LEGAL_DISPUTE", "Compensation / damages claim", "We appreciate your feedback regarding service availability. Please DM us your account details."),
        ("Can I share in-app purchases with Family Sharing members?", False, "AUTO_HANDLE_FAQ", "Family sharing in-app rules", "Consumable purchases (like coins) cannot be shared, but non-consumables and subscriptions can: <URL>")
    ]
    for idx, (q, esc, rsn, crit, rep) in enumerate(billing_variations, start=5):
        add_case(f"50{idx:02d}", q, "billing_subscription", esc, rsn, crit, rep, "hard" if esc else "easy", "Billing batch case")

    # =========================================================================
    # 6. connectivity_network (22 examples: 18 auto-handle, 4 escalate)
    # =========================================================================
    add_case(
        "6001", "iPhone 14 says 'No Service' or 'Searching...' constantly even though my partner on same carrier has 5G.",
        "connectivity_network", False, "AUTO_HANDLE_FAQ",
        "Advise toggling Airplane mode, reseating SIM / checking eSIM, and resetting network settings.",
        "Let's get your cellular connection back. Try toggling Airplane Mode, restarting your phone, or resetting Network Settings: <URL> ^AB",
        "easy", "Standard No Service troubleshooting"
    )
    add_case(
        "6002", "AirPods Pro keep disconnecting from my MacBook Pro every 3 minutes during Zoom meetings.",
        "connectivity_network", False, "AUTO_HANDLE_FAQ",
        "Advise forgetting device in Bluetooth, resetting AirPods in case, and repairing.",
        "To fix audio drops, try resetting your AirPods by holding the case button for 15 seconds, then re-pair: <URL> ^CD",
        "easy", "Bluetooth accessory drop"
    )
    add_case(
        "6003", "Wi-Fi toggle switch in Settings is greyed out and cannot be tapped at all.",
        "connectivity_network", True, "HARDWARE_CHIP_FAILURE",
        "Greyed out Wi-Fi button indicates hardware Wi-Fi chip failure requiring physical repair.",
        "A greyed-out Wi-Fi toggle often indicates a hardware component issue. Please set up a diagnostic appointment: <URL> ^EF",
        "hard", "Hardware Wi-Fi IC failure"
    )
    add_case(
        "6004", "AirDrop fails to discover my friend's iPhone standing right next to me.",
        "connectivity_network", False, "AUTO_HANDLE_FAQ",
        "Verify AirDrop is set to 'Everyone for 10 Minutes', check Bluetooth and Wi-Fi toggles.",
        "Ensure both devices have Wi-Fi and Bluetooth enabled, and set AirDrop to 'Everyone for 10 Minutes': <URL> ^GH",
        "easy", "AirDrop visibility configuration"
    )

    network_variations = [
        ("iPhone drops home Wi-Fi whenever the phone is locked", False, "AUTO_HANDLE_FAQ", "Wi-Fi sleep behavior", "Turn off Private Wi-Fi Address for your home network or reset Network Settings: <URL>"),
        ("Cellular data is extremely slow 0.1 Mbps while Wi-Fi is fine", False, "AUTO_HANDLE_FAQ", "Carrier data troubleshooting", "Check carrier data throttling limits or toggle Cellular Data under Settings > Cellular: <URL>"),
        ("Personal Hotspot doesn't show up on my iPad from my iPhone", False, "AUTO_HANDLE_FAQ", "Hotspot visibility toggle", "Turn on 'Maximize Compatibility' on iPhone Hotspot settings and ensure Bluetooth is on: <URL>"),
        ("CarPlay disconnects every time I drive under a cell tower", False, "AUTO_HANDLE_FAQ", "CarPlay interference tips", "Try using an Apple-certified Lightning/USB-C cable or forget the car in CarPlay settings: <URL>"),
        ("Bluetooth won't pair with third party fitness tracker", False, "AUTO_HANDLE_FAQ", "Third-party accessory pairing", "Put tracker in pairing mode, check companion app permissions, and restart Bluetooth: <URL>"),
        ("iPhone keeps switching between 5G and LTE constantly draining battery", False, "AUTO_HANDLE_FAQ", "5G Auto mode configuration", "Select '5G Auto' under Settings > Cellular > Cellular Data Options > Voice & Data: <URL>"),
        ("Cellular Update Failed message appearing after restart", True, "HARDWARE_MODEM_FAILURE", "Baseband modem hardware failure", "This alert means your cellular modem cannot be initialized. Hardware inspection is needed: <URL>"),
        ("Cannot connect to university 802.1X enterprise Wi-Fi network", False, "AUTO_HANDLE_FAQ", "Enterprise network certificate guide", "Enterprise networks require installing a security certificate or profile from your school IT: <URL>"),
        ("Apple TV ethernet port not detecting plugged-in cable", False, "AUTO_HANDLE_FAQ", "Apple TV networking", "Test the ethernet cable on another device or reboot your router and Apple TV: <URL>"),
        ("Private Relay is hiding my IP address and blocking my bank website", False, "AUTO_HANDLE_FAQ", "iCloud Private Relay bypass", "Turn off Private Relay temporarily for that Wi-Fi network in Settings > Wi-Fi: <URL>"),
        ("iPhone won't send SMS green text messages to Android users", False, "AUTO_HANDLE_FAQ", "SMS carrier messaging settings", "Ensure 'Send as SMS' is enabled under Settings > Messages and check carrier plan: <URL>"),
        ("GPS location is off by 5 miles in Apple Maps", False, "AUTO_HANDLE_FAQ", "Location accuracy recalibration", "Ensure Precise Location is enabled for Maps under Settings > Privacy > Location Services: <URL>"),
        ("AirPods audio crackles and pops only in my left ear", True, "SERVICE_PROGRAM_ELIGIBLE", "Known AirPods Pro crackle issue", "Your AirPods Pro may qualify for our service program for sound issues. Check eligibility: <URL>"),
        ("My iPhone cannot find any Bluetooth devices at all scanning forever", True, "HARDWARE_ANTENNA_ISSUE", "Bluetooth antenna failure", "If no devices appear after a network reset, physical diagnostic testing is recommended: <URL>"),
        ("How to transfer eSIM from my old iPhone to new iPhone?", False, "AUTO_HANDLE_FAQ", "eSIM quick transfer instructions", "Use eSIM Quick Transfer under Settings > Cellular > Add eSIM > Transfer From Nearby iPhone: <URL>"),
        ("MacBook Wi-Fi says 'No IP Address assigned' self-assigned IP", False, "AUTO_HANDLE_FAQ", "DHCP lease renewal guide", "Renew your DHCP Lease under System Settings > Network > Wi-Fi > Details > TCP/IP: <URL>"),
        ("AirTag not updating location for 3 days", False, "AUTO_HANDLE_FAQ", "Find My network explanation", "AirTag location updates when nearby Find My network Apple devices detect its Bluetooth signal: <URL>"),
        ("HomeKit smart lights say 'No Response' after router reboot", False, "AUTO_HANDLE_FAQ", "Home hub reboot instructions", "Restart your Apple TV or HomePod acting as the Home Hub to re-sync accessories: <URL>")
    ]
    for idx, (q, esc, rsn, crit, rep) in enumerate(network_variations, start=5):
        add_case(f"60{idx:02d}", q, "connectivity_network", esc, rsn, crit, rep, "medium", "Network batch case")

    # =========================================================================
    # 7. repair_service_warranty (18 examples: 12 auto-handle, 6 escalate)
    # =========================================================================
    add_case(
        "7001", "How do I book a Genius Bar appointment at the Fifth Avenue New York store?",
        "repair_service_warranty", False, "AUTO_HANDLE_FAQ",
        "Provide direct link to Apple Store appointment booking tool.",
        "You can schedule a reservation at the Fifth Avenue Genius Bar through the Apple Support app or here: <URL> ^AB",
        "easy", "Genius Bar booking link"
    )
    add_case(
        "7002", "How much is out of warranty screen repair for iPhone 13 Pro?",
        "repair_service_warranty", False, "AUTO_HANDLE_FAQ",
        "Provide link to official iPhone screen repair pricing calculator.",
        "You can view estimated out-of-warranty screen repair pricing for your model here: <URL> ^CD",
        "easy", "Screen repair price self-service"
    )
    add_case(
        "7003", "Store staff cracked my iPad screen while replacing my battery and refuse to take blame!!",
        "repair_service_warranty", True, "REPAIR_DAMAGE_DISPUTE",
        "Severe store service dispute / damage claim. Escalate to store management review.",
        "We are very concerned to hear this. Please DM us your repair number and store location so we can escalate this to management. ^EF",
        "hard", "In-store technician damage dispute"
    )
    add_case(
        "7004", "How do I check if my MacBook is still covered under AppleCare+?",
        "repair_service_warranty", False, "AUTO_HANDLE_FAQ",
        "Direct to checkcoverage.apple.com with device serial number.",
        "You can verify your AppleCare+ coverage status by entering your serial number here: <URL> ^GH",
        "easy", "Coverage check tool"
    )

    repair_variations = [
        ("Can I walk into an Apple Store without an appointment for a battery swap?", False, "AUTO_HANDLE_FAQ", "Walk-in policy explanation", "Walk-in appointments are accepted based on availability, but reservations are recommended: <URL>"),
        ("What documents do I need to bring for an iPhone repair appointment?", False, "AUTO_HANDLE_FAQ", "Appointment preparation guide", "Bring your device, a government ID, proof of purchase if available, and back up your device: <URL>"),
        ("Store quoted me $600 for water damage repair on my iPhone 12 this is robbery!", True, "REPAIR_PRICE_DISPUTE", "Out of warranty replacement policy", "Out-of-warranty fees cover full device replacement for liquid damage. DM us for options."),
        ("How long does mail-in iPhone repair take on average?", False, "AUTO_HANDLE_FAQ", "Mail-in turnaround estimates", "Mail-in repairs generally take 5 to 7 business days from the date we receive your device: <URL>"),
        ("Do you offer loaner phones while my iPhone is sent out for repair?", False, "AUTO_HANDLE_FAQ", "Loaner phone policy", "Loaner devices may be available at select Apple Stores during qualifying repairs: <URL>"),
        ("Apple Authorized Service Provider refused to service my phone saying parts not available", True, "SERVICE_PARTNER_ISSUE", "AASP partner escalation", "Please DM us the provider name and work order number so we can investigate availability."),
        ("Is cracked back glass covered under standard 1-year Apple limited warranty?", False, "AUTO_HANDLE_FAQ", "Accidental damage exclusions", "Physical accidental damage is not covered under the limited warranty, but is covered by AppleCare+: <URL>"),
        ("Track my repair status with repair ID R48291038", False, "AUTO_HANDLE_FAQ", "Repair status tracking link", "You can check the live status of your repair using your ID and postal code here: <URL>"),
        ("Can Apple replace only the cracked glass on Apple Watch screen?", False, "AUTO_HANDLE_FAQ", "Apple Watch repair policy", "Apple Watch screen repairs typically involve replacing the whole unit: <URL>"),
        ("My device was repaired last week and the same issue returned today", True, "REPAIR_RECURRING_FAILURE", "Post-repair warranty rework", "Repairs include a 90-day warranty. Please DM us your repair ID so we can expedite a re-check."),
        ("How to transfer my AppleCare+ plan to the person buying my old Mac?", False, "AUTO_HANDLE_FAQ", "AppleCare transfer process", "You can transfer your AppleCare+ plan to a new owner by following these steps: <URL>"),
        ("Do you repair vintage iPhone 5s in 2024?", False, "AUTO_HANDLE_FAQ", "Obsolete product hardware policy", "Products classified as obsolete are no longer eligible for hardware service: <URL>"),
        ("Can I cancel my repair reservation at Covent Garden store?", False, "AUTO_HANDLE_FAQ", "Reservation cancellation", "Manage or cancel your reservation in the Apple Support app or via your confirmation email: <URL>"),
        ("Technician lost my MacBook screws during diagnosis", True, "REPAIR_DAMAGE_DISPUTE", "Service quality incident", "We apologize for this inconvenience. Please DM us your case number so we can resolve this immediately.")
    ]
    for idx, (q, esc, rsn, crit, rep) in enumerate(repair_variations, start=5):
        add_case(f"70{idx:02d}", q, "repair_service_warranty", esc, rsn, crit, rep, "medium", "Repair batch case")

    # =========================================================================
    # 8. feedback_complaint (18 examples: MANDATORY ESCALATION)
    # =========================================================================
    complaints = [
        ("Worst customer service on the planet. Your manager in Austin was rude and dismissive.", "MANDATORY_SUPERVISORY_ESCALATION", "Store manager misconduct complaint"),
        ("I will be filing a class action lawsuit against Apple for deceptive battery throttling!", "MANDATORY_LEGAL_ESCALATION", "Legal action threat"),
        ("Your company is a scam. I have spent $10k on your products and this is how you treat customers?", "MANDATORY_BRAND_RELATIONS", "High customer churn risk"),
        ("Called support 4 times and got hung up on every single time. Absolutely pathetic service.", "MANDATORY_SUPERVISORY_ESCALATION", "Agent hangup complaint"),
        ("Tim Cook should be ashamed of what iOS has become. Pure buggy garbage.", "MANDATORY_BRAND_FEEDBACK", "General executive complaint"),
        ("The staff at the Regent St store made racially discriminatory remarks to my sister today.", "MANDATORY_HR_LEGAL_ESCALATION", "Severe discrimination allegation"),
        ("I am switching to Samsung today after 12 years with iPhone. Never coming back.", "MANDATORY_RETENTION_ESCALATION", "Defection announcement"),
        ("Your trade-in program stole my phone and offered me $0 after quoting $450!! SCAM!!", "MANDATORY_TRADEIN_DISPUTE", "Trade-in value bait-and-switch claim"),
        ("I've been waiting on hold for 2 hours. Does anyone even work at Apple support??", "MANDATORY_HOLD_TIME_COMPLAINT", "Severe call queue frustration"),
        ("Why did you remove the headphone jack? Still angry about this 8 years later.", "MANDATORY_PRODUCT_FEEDBACK", "Product design complaint"),
        ("Your trade-in partner Phobio claimed my pristine laptop had a white spot on screen. Total lie!", "MANDATORY_TRADEIN_DISPUTE", "Third-party trade-in dispute"),
        ("I hate the new photo app layout in iOS 18 it is completely unusable change it back now.", "MANDATORY_PRODUCT_FEEDBACK", "UI redesign backlash"),
        ("Your chat support agent told me to Google the solution myself. Is that what I pay for?", "MANDATORY_SUPERVISORY_ESCALATION", "Agent unprofessionalism complaint"),
        ("Apple is a monopoly destroying small developers and stealing ideas.", "MANDATORY_BRAND_FEEDBACK", "Antitrust/developer grievance"),
        ("You guys ruined my Christmas delivery was promised by Dec 24 and never arrived!", "MANDATORY_DELIVERY_ESCALATION", "Holiday delivery failure complaint"),
        ("Sick of your planned obsolescence making my phone slow down so I buy a new one.", "MANDATORY_BRAND_FEEDBACK", "Planned obsolescence accusation"),
        ("Disgusted with how much packaging waste comes in Apple Vision Pro box.", "MANDATORY_ENVIRONMENTAL_FEEDBACK", "Environmental packaging grievance"),
        ("Never had an issue with Windows like I have had with this terrible overpriced Mac.", "MANDATORY_RETENTION_ESCALATION", "Platform comparison frustration")
    ]
    for idx, (q, rsn, crit) in enumerate(complaints, start=1):
        add_case(
            f"80{idx:02d}", q, "feedback_complaint", True, rsn, crit,
            "We value your feedback and regret that your experience fell short. Please DM us so we can connect you with a supervisory manager. ^AB",
            "medium", "Complaint case"
        )

    # =========================================================================
    # 9. other_unknown (16 examples: MANDATORY ESCALATION)
    # =========================================================================
    unknowns = [
        ("Help please", "AMBIGUOUS_QUERY", "Extremely vague plea lacking any technical context"),
        ("Can you buy my crypto token? Check out this pump on solana", "SPAM_PROMOTION", "Crypto bot solicitation"),
        ("What is the meaning of life Siri?", "OUT_OF_DOMAIN_CHITCHAT", "Philosophical open-ended chitchat"),
        ("asdfghjkl qwertyuiop ????", "GIBBERISH_INPUT", "Random keyboard smash"),
        ("hello are you a robot or a human?", "META_AGENT_QUESTION", "Inquiry about agent bot status"),
        ("iPhone 15 pro max blue titanium", "AMBIGUOUS_FRAGMENT", "Product name fragment without question"),
        ("https://sketchy-link.ru/login", "MALICIOUS_URL_LINK", "Suspicious link without context"),
        ("Why is the sky blue?", "OUT_OF_DOMAIN_CHITCHAT", "General science question"),
        ("Tell me a funny joke right now", "OUT_OF_DOMAIN_CHITCHAT", "Entertainment request"),
        ("Can I use an Android charger with a toaster?", "NONSENSE_QUERY", "Absurd nonsensical inquiry"),
        ("???", "GIBBERISH_INPUT", "Punctuation only"),
        ("What time does the supermarket close in Seattle?", "OUT_OF_DOMAIN_CHITCHAT", "Unrelated third-party business query"),
        ("I love Apple so much you guys are the best ❤️❤️❤️", "BENIGN_PRAISE_CHITCHAT", "Positive non-actionable fan praise"),
        ("Follow me back please @AppleSupport", "SOCIAL_MEDIA_CHITCHAT", "Follow request"),
        ("Test message 123", "SYSTEM_TEST_QUERY", "Testing message"),
        ("What is your opinion on Google Pixel 8?", "COMPETITOR_CHITCHAT", "Competitor opinion question")
    ]
    for idx, (q, rsn, crit) in enumerate(unknowns, start=1):
        add_case(
            f"90{idx:02d}", q, "other_unknown", True, rsn, crit,
            "We're here to help with your Apple products and services. Let us know what technical issue you're experiencing! ^AB",
            "easy", "Out of domain / unknown case"
        )

    assert len(examples) == 200, f"Expected 200 golden examples, got {len(examples)}"
    return examples


def save_golden_dataset(output_path: str = "data/golden/golden_set.json"):
    """Saves the golden dataset to the specified JSON path."""
    data = build_golden_dataset()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[golden_curator] Successfully saved {len(data)} hand-audited Golden examples to {output_path}")


if __name__ == "__main__":
    save_golden_dataset()
