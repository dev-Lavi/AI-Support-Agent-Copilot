"""Module for reconstructing customer-brand support conversation pairs from raw tweets."""

from typing import Dict, List, Optional
import pandas as pd
from src.data.cleaner import clean_tweet_text, is_valid_tweet


def reconstruct_conversation_pairs(
    df: pd.DataFrame,
    brand_handle: str = "AppleSupport",
    max_pairs: Optional[int] = None,
) -> pd.DataFrame:
    """Reconstructs customer -> brand reply pairs from the raw Twitter dataset.

    Args:
        df: Raw DataFrame with columns:
            tweet_id, author_id, inbound, created_at, text,
            response_tweet_id, in_response_to_tweet_id
        brand_handle: Twitter handle of the brand (case-insensitive)
        max_pairs: Optional cap on the number of returned pairs

    Returns:
        DataFrame with columns:
            pair_id, conversation_id, customer_tweet_id, customer_text,
            brand_tweet_id, brand_text, created_at, brand_handle
    """
    # Normalize author_id and handle
    brand_handle_lower = brand_handle.lower()
    df["author_id_str"] = df["author_id"].astype(str).str.lower()

    # Identify brand replies that are in response to a prior tweet
    brand_replies = df[
        (df["author_id_str"] == brand_handle_lower)
        & (~df["inbound"].astype(bool))
        & (df["in_response_to_tweet_id"].notna())
    ].copy()

    # Create lookup map for parent customer tweets: tweet_id -> row
    # In twcs.csv, tweet_id can be numeric or string
    tweet_lookup: Dict[str, dict] = {}
    for _, row in df[df["inbound"].astype(bool)].iterrows():
        t_id = str(row["tweet_id"]).strip()
        tweet_lookup[t_id] = {
            "customer_text": str(row["text"]),
            "customer_tweet_id": t_id,
            "created_at": row.get("created_at", ""),
        }

    pairs: List[dict] = []
    seen_customer_texts = set()

    for _, b_row in brand_replies.iterrows():
        parent_id = str(b_row["in_response_to_tweet_id"]).strip()
        # Some in_response_to_tweet_id contain decimal from float conversion (.0)
        if parent_id.endswith(".0"):
            parent_id = parent_id[:-2]

        if parent_id in tweet_lookup:
            c_info = tweet_lookup[parent_id]
            raw_c_text = c_info["customer_text"]
            raw_b_text = str(b_row["text"])

            clean_c = clean_tweet_text(raw_c_text, preserve_brand_mention=brand_handle)
            clean_b = clean_tweet_text(raw_b_text, preserve_brand_mention=brand_handle)

            if is_valid_tweet(clean_c) and is_valid_tweet(clean_b):
                # Simple exact deduplication on customer query
                c_hash = clean_c.lower()
                if c_hash in seen_customer_texts:
                    continue
                seen_customer_texts.add(c_hash)

                pair_id = f"pair_{len(pairs):06d}"
                conv_id = f"conv_{parent_id}"

                pairs.append({
                    "pair_id": pair_id,
                    "conversation_id": conv_id,
                    "customer_tweet_id": parent_id,
                    "customer_text": clean_c,
                    "raw_customer_text": raw_c_text,
                    "brand_tweet_id": str(b_row["tweet_id"]),
                    "brand_text": clean_b,
                    "raw_brand_text": raw_b_text,
                    "created_at": c_info["created_at"],
                    "brand_handle": brand_handle,
                })

                if max_pairs and len(pairs) >= max_pairs:
                    break

    return pd.DataFrame(pairs)
