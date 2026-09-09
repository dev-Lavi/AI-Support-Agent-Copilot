"""Script to display final benchmark results and headline metrics in the terminal.

Usage:
    python scripts/show_results.py
"""

import json
import sys
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    metrics_dir = PROJECT_ROOT / "results/metrics"
    comp_file = metrics_dir / "baseline_comparison.json"
    judge_file = metrics_dir / "judge_calibration.json"
    esc_file = metrics_dir / "escalation_metrics.json"

    if not comp_file.exists():
        print(f"[show_results] Results not found at {comp_file}. Please run 'python scripts/evaluate.py' first.")
        return

    with open(comp_file, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    print("\n" + "=" * 80)
    print("           HIVERA AI SUPPORT AGENT — HEADLINE BENCHMARK RESULTS")
    print("=" * 80)

    headers = ["System", "Accuracy", "Macro F1", "Coverage", "FAHR (Safety)", "Esc. Recall", "Groundedness"]
    rows = [
        [
            item["System"],
            item["Intent Accuracy"],
            item["Intent Macro F1"],
            item["Auto-Handle Coverage"],
            item["False Auto-Handle Rate (FAHR)"],
            item["Escalation Recall"],
            item["Groundedness (1-5)"]
        ]
        for item in comp_data
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print("=" * 80)

    if judge_file.exists():
        with open(judge_file, "r", encoding="utf-8") as f:
            j_data = json.load(f)
        print("\n[LLM-as-Judge Human Agreement Calibration]")
        print(f" • Spearman Rank Correlation (rho): {j_data.get('spearman_rho', 'N/A')}")
        print(f" • Pearson Correlation (r):         {j_data.get('pearson_r', 'N/A')}")
        print(f" • Cohen's Quadratic Kappa:         {j_data.get('cohens_quadratic_kappa', 'N/A')}")
        print(f" • Judge Bias (Mean Error):          {j_data.get('judge_bias', 'N/A')}")
        print(f" • Status:                           {j_data.get('acceptance_status', 'N/A')}")

    if esc_file.exists():
        with open(esc_file, "r", encoding="utf-8") as f:
            esc_data = json.load(f)
        agent_esc = esc_data.get("final_agent", {})
        print("\n[Final Agent Safety & Escalation Profile]")
        print(f" • Total Queries Tested:             {agent_esc.get('total_evaluated', 'N/A')}")
        print(f" • Auto-Handled:                     {agent_esc.get('n_auto_handled', 'N/A')} ({agent_esc.get('coverage', 0)*100:.1f}%)")
        print(f" • Escalated to Humans:              {agent_esc.get('n_escalated', 'N/A')}")
        print(f" • False Auto-Handle Rate (FAHR):    {agent_esc.get('false_auto_handle_rate', 0)*100:.1f}% (Safety Goal < 5%)")
        print(f" • Escalation Precision:             {agent_esc.get('escalation_precision', 0)*100:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
