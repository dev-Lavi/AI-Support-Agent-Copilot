"""Interactive and single-query execution CLI for the AI Customer Support Agent.

Usage:
    python scripts/run_agent.py --query "My iPhone battery dies within 2 hours after updating"
    python scripts/run_agent.py --interactive
"""

import argparse
import sys
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.agent import SupportAgentPipeline


def print_triage_result(res):
    """Prints formatted triage result for a query."""
    print("\n" + "=" * 75)
    print(f"CUSTOMER QUERY: \"{res.query}\"")
    print("=" * 75)
    print(f" • Predicted Intent:     {res.predicted_intent.upper()} (Confidence: {res.intent_confidence*100:.1f}%)")
    print(f" • Retrieval Match:      Cosine Similarity {res.retrieval_similarity:.3f}")

    dec_color = "AUTO-HANDLE" if res.decision == "AUTO_HANDLE" else "ESCALATE TO HUMAN"
    print(f" • Escalation Decision:  [{dec_color}]")
    print(f" • Reason Code:          {res.reason_code}")
    print(f" • Rationale:            {res.reason_details}")
    print("-" * 75)
    print(f" • Grounded Reply Draft: \"{res.draft_reply}\"")
    print(f" • Groundedness Rating:  {res.groundedness_score} / 5.0")

    if res.retrieved_evidence:
        print("\n[Retrieved Historical Precedent Evidence]")
        top = res.retrieved_evidence[0]
        print(f"   Historical Query: \"{top['historical_customer_query']}\"")
        print(f"   Historical Reply: \"{top['historical_brand_reply']}\"")
        print(f"   Similarity:       {top['similarity_score']:.4f}")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run the AI Support Agent.")
    parser.add_argument("--query", type=str, help="Single customer tweet query")
    parser.add_argument("--interactive", action="store_true", help="Start interactive CLI loop")
    parser.add_argument("--models-dir", type=str, default="results/models", help="Models directory")
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / args.models_dir
    if not (models_dir / "intent_classifier.pkl").exists():
        print(f"[run_agent] Trained model not found at {models_dir}. Running training first...")
        from scripts.train_models import main as run_train
        run_train()

    print("[run_agent] Loading trained agent pipeline...")
    pipeline = SupportAgentPipeline.load(str(models_dir))

    if args.interactive:
        print("\n--- AI Customer Support Agent Interactive Session (@AppleSupport) ---")
        print("Type your customer support tweet below. (Type 'exit' or 'quit' to stop)\n")
        while True:
            try:
                user_input = input("Customer Tweet > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Exiting. Goodbye!")
                    break
                result = pipeline.process(user_input)
                print_triage_result(result)
            except (KeyboardInterrupt, EOFError):
                break
    elif args.query:
        result = pipeline.process(args.query)
        print_triage_result(result)
    else:
        # Default demo query
        demo_q = "My iPhone battery dies within 2 hours after updating to iOS 17. Help!"
        print(f"[run_agent] No query specified. Running default demo query:\n> \"{demo_q}\"")
        result = pipeline.process(demo_q)
        print_triage_result(result)


if __name__ == "__main__":
    main()
