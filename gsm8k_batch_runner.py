#!/usr/bin/env python
"""
Run the GSM8K benchmark multiple times and save each run separately.

Usage:
  python AI525_Project/gsm8k_batch_runner.py --runs 10 --limit 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from gsm8k_visualize import load_jsonl, write_stats_csv


def compute_stats(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float, float, float]:
    """Compute aggregate stats and return (stats dict, improvement, baseline_acc, agentic_acc)."""
    total = len(rows)
    if total == 0:
        raise ValueError("No rows found in results JSONL; cannot compute stats.")

    baseline_correct = sum(1 for r in rows if r.get("baseline_correct"))
    agentic_correct = sum(1 for r in rows if r.get("agentic_correct"))
    corrected = sum(
        1 for r in rows if (not r.get("baseline_correct")) and r.get("agentic_correct")
    )
    degraded = sum(
        1 for r in rows if r.get("baseline_correct") and (not r.get("agentic_correct"))
    )
    unchanged_wrong = sum(
        1
        for r in rows
        if (not r.get("baseline_correct")) and (not r.get("agentic_correct"))
    )

    baseline_acc = baseline_correct / total
    agentic_acc = agentic_correct / total
    accuracy_improvement = agentic_acc - baseline_acc

    stats = {
        "total": total,
        "baseline_correct": baseline_correct,
        "agentic_correct": agentic_correct,
        "corrected_by_verifier": corrected,
        "degraded_by_verifier": degraded,
        "unchanged_wrong": unchanged_wrong,
        "baseline_accuracy": f"{baseline_acc:.3f}",
        "agentic_accuracy": f"{agentic_acc:.3f}",
        "accuracy_improvement": f"{accuracy_improvement:.3f}",
    }
    return stats, accuracy_improvement, baseline_acc, agentic_acc


def run_single_benchmark(
    run_idx: int, limit: int, project_dir: Path, output_dir: Path
) -> Dict[str, Any]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = output_dir / f"gsm8k_results_run{run_idx}_{timestamp}.jsonl"
    summary_path = output_dir / f"gsm8k_summary_run{run_idx}_{timestamp}.csv"

    cmd = [
        sys.executable,
        str(project_dir / "gsm8k_benchmark.py"),
        "--limit",
        str(limit),
        "--output",
        str(results_path),
        "--summary-csv",
        str(summary_path),
    ]
    print(f"[Run {run_idx}] Running benchmark: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    rows = load_jsonl(results_path)
    stats, improvement, baseline_acc, agentic_acc = compute_stats(rows)
    improvement_tag = f"{improvement:+.3f}"
    stats_path = output_dir / f"gsm8k_stats_{improvement_tag}.csv"
    write_stats_csv(stats_path, stats)

    print(
        f"[Run {run_idx}] baseline_acc={baseline_acc:.3f} "
        f"agentic_acc={agentic_acc:.3f} diff={improvement_tag}"
    )
    print(f"[Run {run_idx}] Results: {results_path}")
    print(f"[Run {run_idx}] Summary CSV: {summary_path}")
    print(f"[Run {run_idx}] Stats CSV: {stats_path}")

    return {
        "run": run_idx,
        "results_path": results_path,
        "summary_path": summary_path,
        "stats_path": stats_path,
        "baseline_accuracy": baseline_acc,
        "agentic_accuracy": agentic_acc,
        "accuracy_improvement": improvement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run gsm8k_benchmark.py multiple times and save each run separately."
    )
    parser.add_argument("--runs", type=int, default=10, help="How many runs to perform.")
    parser.add_argument("--limit", type=int, default=50, help="Question limit per run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gsm8k_results/batch_runs"),
        help="Directory to store per-run outputs.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries = []
    for run_idx in range(1, args.runs + 1):
        run_summary = run_single_benchmark(run_idx, args.limit, project_dir, output_dir)
        run_summaries.append(run_summary)

    print("\nAll runs completed:")
    for summary in run_summaries:
        tag = f"{summary['accuracy_improvement']:+.3f}"
        print(
            f"Run {summary['run']}: baseline={summary['baseline_accuracy']:.3f} "
            f"agentic={summary['agentic_accuracy']:.3f} diff={tag} -> {summary['stats_path']}"
        )


if __name__ == "__main__":
    main()
