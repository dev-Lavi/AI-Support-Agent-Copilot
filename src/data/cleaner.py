"""Tweet text sanitization, normalization, and entity masking utilities."""

import html
import re
from typing import Optional

# Regex patterns
URL_REGEX = re.compile(r"https?://\S+|www\.\S+|http://t\.co/\S+")
MENTION_REGEX = re.compile(r"@\w+")
WHITESPACE_REGEX = re.compile(r"\s+")
MULTIPLE_PUNCT_REGEX = re.compile(r"([!?.]){2,}")


def clean_tweet_text(
    text: Optional[str],
    mask_urls: bool = True,
    mask_mentions: bool = True,
    preserve_brand_mention: Optional[str] = "AppleSupport",
) -> str:
    """Cleans raw tweet text by unescaping HTML, normalizing whitespace,

    and selectively masking URLs and user handles while preserving critical semantics.

    Args:
        text: Raw tweet text string.
        mask_urls: Whether to replace URLs with '<URL>'.
        mask_mentions: Whether to replace user handles with '<USER>'.
        preserve_brand_mention: Specific brand handle to mask as '<BRAND>' instead of '<USER>'.

    Returns:
        Sanitized text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (e.g. &amp; -> &, &lt; -> <)
    cleaned = html.unescape(text)

    # 2. Mask URLs
    if mask_urls:
        cleaned = URL_REGEX.sub("<URL>", cleaned)

    # 3. Handle mentions
    if mask_mentions:
        if preserve_brand_mention:
            brand_pattern = re.compile(rf"@{preserve_brand_mention}\b", re.IGNORECASE)
            cleaned = brand_pattern.sub("<BRAND>", cleaned)
        cleaned = MENTION_REGEX.sub("<USER>", cleaned)

    # 4. Normalize multiple punctuation (e.g. "???" -> "??", "!!!!" -> "!!")
    cleaned = MULTIPLE_PUNCT_REGEX.sub(r"\1\1", cleaned)

    # 5. Normalize whitespace (remove newlines, tabs, double spaces)
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()

    return cleaned


def is_valid_tweet(text: str, min_chars: int = 10, max_chars: int = 500) -> bool:
    """Checks if a tweet contains meaningful textual content for customer support."""
    if not text or len(text.strip()) < min_chars or len(text) > max_chars:
        return False

    # Check if text contains more than just tokens
    stripped = re.sub(r"<URL>|<USER>|<BRAND>", "", text).strip()
    return len(stripped) >= 5
