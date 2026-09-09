"""Grounded Reply Generator and Evidence-Conditioned Drafting Engine."""

import re
from typing import List, Dict, Tuple, Optional


class GroundedReplyDrafter:
    """Drafts customer support replies strictly grounded in retrieved historical brand resolutions."""

    def __init__(self, brand_handle: str = "AppleSupport"):
        self.brand_handle = brand_handle

    def draft_reply(
        self,
        query: str,
        predicted_intent: str,
        retrieved_evidence: List[Tuple[Dict, float]]
    ) -> Dict:
        """Synthesizes a grounded reply from the top retrieved historical precedent.

        Args:
            query: Cleaned incoming customer query.
            predicted_intent: Classified intent label.
            retrieved_evidence: List of (historical_doc, similarity_score) tuples.

        Returns:
            Dict containing draft_reply, groundedness_score, extracted_urls, and citation_doc_id.
        """
        if not retrieved_evidence:
            return {
                "draft_reply": f"We're here to help! Please DM us so we can take a closer look at your issue with your Apple device. ^{self._get_signoff()}",
                "groundedness_score": 1.0,
                "extracted_urls": [],
                "citation_doc_id": None
            }

        top_doc, top_sim = retrieved_evidence[0]
        hist_reply = top_doc.get("brand_text", "")

        # Extract verified URLs from historical precedent
        urls = re.findall(r"<URL>", hist_reply)

        # Build grounded response based on the top historical resolution
        # Clean up any duplicated whitespace
        grounded_text = hist_reply.strip()

        # Ensure Twitter character limit compliance (<= 280 chars)
        if len(grounded_text) > 280:
            grounded_text = grounded_text[:275] + "..."

        # Calculate estimated groundedness rating (1 to 5 scale)
        # Higher retrieval similarity -> stronger grounding
        if top_sim >= 0.80:
            groundedness = 5.0
        elif top_sim >= 0.70:
            groundedness = 4.5
        elif top_sim >= 0.60:
            groundedness = 3.5
        elif top_sim >= 0.50:
            groundedness = 2.5
        else:
            groundedness = 1.5

        return {
            "draft_reply": grounded_text,
            "groundedness_score": groundedness,
            "extracted_urls": urls,
            "citation_doc_id": top_doc.get("pair_id", "unknown"),
            "historical_query_matched": top_doc.get("customer_text", ""),
            "retrieval_similarity": round(top_sim, 4)
        }

    def _get_signoff(self) -> str:
        return "AB"
