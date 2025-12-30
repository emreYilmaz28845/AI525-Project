#!/usr/bin/env python
# coding: utf-8

"""
Visualize GSM8K benchmark results.

Example:
  python AI525_Project/gsm8k_visualize.py --input AI525_Project/gsm8k_results.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional visualization
    plt = None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_stats_csv(path: Path, stats: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("AI525_Project/gsm8k_results.jsonl"),
        help="Input JSONL path from gsm8k_benchmark.py.",
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=Path("AI525_Project/gsm8k_stats.csv"),
        help="Output CSV for aggregate stats.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("AI525_Project/gsm8k_plots.png"),
        help="Output PNG for plots.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    total = len(rows)
    if total == 0:
        print("No rows found.")
        return

    baseline_correct = sum(1 for r in rows if r.get("baseline_correct"))
    agentic_correct = sum(1 for r in rows if r.get("agentic_correct"))
    corrected = sum(
        1
        for r in rows
        if (not r.get("baseline_correct")) and r.get("agentic_correct")
    )
    degraded = sum(
        1
        for r in rows
        if r.get("baseline_correct") and (not r.get("agentic_correct"))
    )
    unchanged_wrong = sum(
        1
        for r in rows
        if (not r.get("baseline_correct")) and (not r.get("agentic_correct"))
    )

    correction_steps = [
        r.get("correction_step")
        for r in rows
        if r.get("correction_step") is not None
    ]
    step_counts = Counter(correction_steps)
    decisions = Counter()
    for r in rows:
        for decision in r.get("verifier_decisions", []) or []:
            decisions[decision] += 1

    stats = {
        "total": total,
        "baseline_correct": baseline_correct,
        "agentic_correct": agentic_correct,
        "corrected_by_verifier": corrected,
        "degraded_by_verifier": degraded,
        "unchanged_wrong": unchanged_wrong,
        "baseline_accuracy": f"{baseline_correct / total:.3f}",
        "agentic_accuracy": f"{agentic_correct / total:.3f}",
        "accuracy_improvement": f"{(agentic_correct - baseline_correct) / total:.3f}",
    }
    write_stats_csv(args.output_stats, stats)
    print(f"Wrote stats CSV: {args.output_stats}")

    if plt is None:
        print("matplotlib not available; skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(
        ["Baseline", "Agentic"], [baseline_correct, agentic_correct], color=["#4C78A8", "#F58518"]
    )
    axes[0, 0].set_title("Correct Answers")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].bar(
        ["Corrected", "Degraded", "Unchanged Wrong"],
        [corrected, degraded, unchanged_wrong],
        color=["#54A24B", "#E45756", "#B279A2"],
    )
    axes[0, 1].set_title("Outcome Changes")
    axes[0, 1].set_ylabel("Count")

    if step_counts:
        steps = sorted(step_counts.keys())
        axes[1, 0].bar([str(s) for s in steps], [step_counts[s] for s in steps])
    axes[1, 0].set_title("Correction Step (1-based)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Count")

    if decisions:
        labels = list(decisions.keys())
        values = [decisions[k] for k in labels]
        axes[1, 1].bar(labels, values)
    axes[1, 1].set_title("Verifier Decisions (All Attempts)")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    print(f"Wrote plots PNG: {args.output_plot}")


if __name__ == "__main__":
    main()
