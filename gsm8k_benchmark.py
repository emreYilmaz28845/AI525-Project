#!/usr/bin/env python
# coding: utf-8

"""
Benchmark baseline vs self-verification on GSM8K (test split).

Example:
  python AI525_Project/gsm8k_benchmark.py --limit 50 --output AI525_Project/gsm8k_results.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset
from langchain_core.messages import AIMessage, HumanMessage

from collections import Counter

from self_verification_agent_local_llama import (
    MAX_ATTEMPTS,
    ANSWER_TOLERANCE,
    CONFIDENCE_THRESHOLD,
    SOLVER_SYSTEM,
    VERIFIER_SYSTEM,
    build_llms,
    build_feedback_message,
    call_verifier_with_retries,
    enforce_verifier_structure,
    answers_are_equivalent,
    is_valid_critique,
    extract_verifier_answer,
    parse_decision,
    strip_confidence,
)

# Number of samples for majority voting
MAJORITY_VOTES = int(os.getenv("MAJORITY_VOTES", "3"))


ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
FINAL_ANSWER_PATTERN = re.compile(
    r"(final answer|answer is|therefore|so the answer is)[:\s]*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass
class EvalRow:
    question: str
    expected_answer: str
    baseline_answer: str
    agentic_answer: str
    baseline_extracted: Optional[str]
    agentic_extracted: Optional[str]
    baseline_correct: bool
    agentic_correct: bool
    initial_correct: bool
    attempt_extracted: list[Optional[str]]
    attempt_correct: list[bool]
    correction_step: Optional[int]
    verifier_decisions: list[str]
    attempts: int
    baseline_time_s: float
    agentic_time_s: float


def load_gsm8k(limit: int) -> list[dict]:
    dataset = load_dataset("gsm8k", "main", split="test")
    if limit > 0:
        dataset = dataset.select(range(limit))
    return list(dataset)


def parse_expected_answer(answer_text: str) -> Optional[str]:
    match = ANSWER_PATTERN.search(answer_text or "")
    return match.group(1) if match else None


def extract_final_number(text: str) -> Optional[str]:
    if not text:
        return None
    match = FINAL_ANSWER_PATTERN.search(text)
    if match:
        return match.group(2)
    numbers = NUMBER_PATTERN.findall(text)
    return numbers[-1] if numbers else None


def normalize_number(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        normalized = str(int(float(value)))
        return normalized
    except ValueError:
        return None


def write_jsonl(path: Path, rows: Iterable[EvalRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, ensure_ascii=True) + "\n")


def write_summary_csv(path: Path, rows: Iterable[EvalRow]) -> None:
    fieldnames = [
        "question",
        "expected_answer",
        "baseline_extracted",
        "agentic_extracted",
        "baseline_correct",
        "agentic_correct",
        "initial_correct",
        "correction_step",
        "verifier_decisions",
        "attempts",
        "baseline_time_s",
        "agentic_time_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question": row.question,
                    "expected_answer": row.expected_answer,
                    "baseline_extracted": row.baseline_extracted,
                    "agentic_extracted": row.agentic_extracted,
                    "baseline_correct": row.baseline_correct,
                    "agentic_correct": row.agentic_correct,
                    "initial_correct": row.initial_correct,
                    "correction_step": row.correction_step,
                    "verifier_decisions": "|".join(row.verifier_decisions),
                    "attempts": row.attempts,
                    "baseline_time_s": f"{row.baseline_time_s:.4f}",
                    "agentic_time_s": f"{row.agentic_time_s:.4f}",
                }
            )


def format_question(question: str) -> str:
    return (
        f"{question}\n\n"
        "Show your reasoning briefly. End with 'Final Answer: <number>'."
    )


def run_baseline(solver, question: str) -> tuple[str, float]:
    start = time.time()
    prompt = format_question(question)
    answer = solver.invoke([SOLVER_SYSTEM, HumanMessage(content=prompt)]).content
    elapsed = time.time() - start
    return answer, elapsed


def run_agentic(
    solver, verifier, question: str
) -> tuple[str, str, list[str], list[str], int, float]:
    history = [HumanMessage(content=question)]
    decisions: list[str] = []
    answers: list[str] = []
    last_answer = ""
    first_answer = ""
    start = time.time()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt == 1:
            confidence_prompt = (
                f"{format_question(question)}\n\n"
                "After your answer, add a separate line 'Confidence: <0-10>' to rate your certainty."
            )
            history = [HumanMessage(content=confidence_prompt)]
        solver_msgs = [SOLVER_SYSTEM] + history
        solver_resp = solver.invoke(solver_msgs)
        cleaned_answer, solver_confidence = strip_confidence(solver_resp.content or "")
        last_answer = cleaned_answer
        answers.append(cleaned_answer)
        history.append(AIMessage(content=cleaned_answer))

        if attempt == 1:
            first_answer = last_answer
            if solver_confidence is not None and solver_confidence >= CONFIDENCE_THRESHOLD:
                print(
                    f"[Agentic] High confidence ({solver_confidence}/10); skipping verification."
                )
                decisions.append("ACCEPT")
                history.append(
                    AIMessage(
                        content=f"DECISION: ACCEPT (confidence {solver_confidence}/10)"
                    )
                )
                break

        verifier_msgs = [VERIFIER_SYSTEM] + history
        critique = call_verifier_with_retries(verifier, verifier_msgs)
        critique = enforce_verifier_structure(verifier, verifier_msgs, critique)
        critique_text = (critique.content or "").strip()
        decision = parse_decision(critique_text)
        if decision == "REVISE":
            # Conservative filtering: only revise if we're confident it's a real error
            if answers_are_equivalent(last_answer, critique_text, tolerance=ANSWER_TOLERANCE):
                decision = "ACCEPT"  # Answers match, no need to revise
            elif not is_valid_critique(critique_text, last_answer):
                decision = "ACCEPT"  # Critique not specific enough, don't risk degradation
            elif attempt > 1:
                # After first revision attempt, stop - further revisions often degrade
                decision = "ACCEPT"
        decisions.append(decision)
        history.append(critique)

        if decision == "ACCEPT":
            break

        history.append(build_feedback_message(critique_text))

    elapsed = time.time() - start
    return last_answer, first_answer, answers, decisions, len(decisions), elapsed


def run_majority_voting(
    solver, question: str, num_votes: int = 3
) -> tuple[str, list[str], list[str], float]:
    """
    Run the solver multiple times with temperature > 0 and take majority vote.
    Returns: (final_answer, all_answers, extracted_numbers, elapsed_time)
    """
    from langchain_ollama import ChatOllama
    from self_verification_agent_local_llama import SOLVER_MODEL, REQUEST_TIMEOUT

    # Use moderate temperature for diversity while maintaining quality
    # 0.7 was too high and caused random errors; 0.3 provides diversity without chaos
    common_kwargs = {"request_timeout": REQUEST_TIMEOUT} if REQUEST_TIMEOUT else {}
    diverse_solver = ChatOllama(model=SOLVER_MODEL, temperature=0.3, **common_kwargs)

    start = time.time()
    prompt = format_question(question)

    answers = []
    extracted_numbers = []

    for i in range(num_votes):
        response = diverse_solver.invoke([SOLVER_SYSTEM, HumanMessage(content=prompt)])
        answer_text = response.content or ""
        answers.append(answer_text)

        # Extract the final number
        extracted = extract_final_number(answer_text)
        if extracted:
            extracted_numbers.append(normalize_number(extracted) or extracted)

    # Find majority answer
    if extracted_numbers:
        counter = Counter(extracted_numbers)
        majority_answer, count = counter.most_common(1)[0]
        # Find the full answer text that corresponds to majority
        for i, num in enumerate(extracted_numbers):
            if num == majority_answer:
                final_answer = answers[i]
                break
        else:
            final_answer = answers[0]
    else:
        final_answer = answers[0] if answers else ""
        majority_answer = None

    elapsed = time.time() - start
    return final_answer, answers, extracted_numbers, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of questions.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gsm8k_results/gsm8k_results.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("gsm8k_results/gsm8k_summary.csv"),
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["verification", "majority", "both"],
        default="both",
        help="Mode: 'verification' (self-verify), 'majority' (majority voting), 'both' (compare all)",
    )
    parser.add_argument(
        "--votes",
        type=int,
        default=3,
        help="Number of votes for majority voting (default: 3)",
    )
    args = parser.parse_args()

    records = load_gsm8k(args.limit)
    solver, verifier = build_llms()

    # Warmup: Load both models into memory before timing starts
    print("[Warmup] Loading solver model...")
    solver.invoke([HumanMessage(content="What is 2+2?")])
    if args.mode in ("verification", "both"):
        print("[Warmup] Loading verifier model...")
        verifier.invoke([HumanMessage(content="What is 2+2?")])
    print(f"[Warmup] Models loaded. Starting benchmark (mode={args.mode}).\n")

    rows: list[EvalRow] = []
    baseline_correct_total = 0
    agentic_correct_total = 0
    majority_correct_total = 0
    correction_total = 0
    false_positive_total = 0
    false_negative_total = 0
    baseline_time_total = 0.0
    agentic_time_total = 0.0
    majority_time_total = 0.0

    for idx, record in enumerate(records, start=1):
        question = record.get("question", "")
        expected = parse_expected_answer(record.get("answer", ""))
        expected_norm = normalize_number(expected)

        print(f"\n[{idx}/{len(records)}] {question[:80]}...")

        # Always run baseline
        baseline_answer, baseline_time = run_baseline(solver, question)
        baseline_extracted = extract_final_number(baseline_answer)
        baseline_norm = normalize_number(baseline_extracted)
        baseline_correct = expected_norm is not None and baseline_norm == expected_norm

        print(
            "[Baseline] extracted="
            f"{baseline_extracted} correct={baseline_correct} time={baseline_time:.2f}s"
        )
        baseline_correct_total += int(baseline_correct)
        baseline_time_total += baseline_time

        # Self-verification mode
        agentic_answer = ""
        agentic_extracted = None
        agentic_correct = False
        first_answer = ""
        first_correct = False
        attempt_answers = []
        decisions = []
        attempts = 0
        agentic_time = 0.0

        if args.mode in ("verification", "both"):
            (
                agentic_answer,
                first_answer,
                attempt_answers,
                decisions,
                attempts,
                agentic_time,
            ) = run_agentic(solver, verifier, question)

            agentic_extracted = extract_final_number(agentic_answer)
            agentic_norm = normalize_number(agentic_extracted)
            first_extracted = extract_final_number(first_answer)
            first_norm = normalize_number(first_extracted)

            agentic_correct = expected_norm is not None and agentic_norm == expected_norm
            first_correct = expected_norm is not None and first_norm == expected_norm

            print(
                "[Verification] extracted="
                f"{agentic_extracted} correct={agentic_correct} attempts={attempts} "
                f"time={agentic_time:.2f}s"
            )

            agentic_correct_total += int(agentic_correct)
            agentic_time_total += agentic_time
            correction_total += int((not baseline_correct) and agentic_correct)

            if decisions:
                final_decision = decisions[-1]
                false_positive_total += int(
                    final_decision == "ACCEPT" and not agentic_correct
                )
                false_negative_total += int(
                    expected_norm is not None
                    and first_norm == expected_norm
                    and decisions[0] == "REVISE"
                )

        # Majority voting mode
        majority_answer = ""
        majority_extracted = None
        majority_correct = False
        vote_answers = []
        vote_numbers = []
        majority_time = 0.0

        if args.mode in ("majority", "both"):
            (
                majority_answer,
                vote_answers,
                vote_numbers,
                majority_time,
            ) = run_majority_voting(solver, question, num_votes=args.votes)

            majority_extracted = extract_final_number(majority_answer)
            majority_norm = normalize_number(majority_extracted)
            majority_correct = expected_norm is not None and majority_norm == expected_norm

            print(
                f"[Majority@{args.votes}] votes={vote_numbers} "
                f"extracted={majority_extracted} correct={majority_correct} "
                f"time={majority_time:.2f}s"
            )

            majority_correct_total += int(majority_correct)
            majority_time_total += majority_time

        # Build row (using verification fields, add majority info to decisions field)
        attempt_extracted = [
            extract_final_number(answer) for answer in attempt_answers
        ] if attempt_answers else []
        attempt_correct = [
            normalize_number(value) == expected_norm if expected_norm is not None else False
            for value in attempt_extracted
        ] if attempt_extracted else []

        correction_step = None
        if expected_norm is not None and attempt_correct:
            for idx_attempt, is_correct in enumerate(attempt_correct, start=1):
                if is_correct:
                    correction_step = idx_attempt
                    break

        rows.append(
            EvalRow(
                question=question,
                expected_answer=expected or "",
                baseline_answer=baseline_answer,
                agentic_answer=agentic_answer if args.mode != "majority" else majority_answer,
                baseline_extracted=baseline_extracted,
                agentic_extracted=agentic_extracted if args.mode != "majority" else majority_extracted,
                baseline_correct=baseline_correct,
                agentic_correct=agentic_correct if args.mode != "majority" else majority_correct,
                initial_correct=first_correct if args.mode in ("verification", "both") else baseline_correct,
                attempt_extracted=attempt_extracted if args.mode != "majority" else vote_numbers,
                attempt_correct=attempt_correct,
                correction_step=correction_step,
                verifier_decisions=decisions if args.mode != "majority" else vote_numbers,
                attempts=attempts if args.mode != "majority" else args.votes,
                baseline_time_s=baseline_time,
                agentic_time_s=agentic_time if args.mode != "majority" else majority_time,
            )
        )

    write_jsonl(args.output, rows)
    if args.summary_csv:
        write_summary_csv(args.summary_csv, rows)

    total = max(len(records), 1)
    baseline_acc = baseline_correct_total / total
    baseline_avg_time = baseline_time_total / total

    print("\n=== Metrics ===")
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    print(f"Avg baseline time (s): {baseline_avg_time:.2f}")

    if args.mode in ("verification", "both"):
        agentic_acc = agentic_correct_total / total
        verification_improvement = agentic_acc - baseline_acc
        correction_rate = correction_total / total
        false_positive_rate = false_positive_total / total
        false_negative_rate = false_negative_total / total
        agentic_avg_time = agentic_time_total / total

        print(f"\n[Self-Verification]")
        print(f"  Accuracy: {agentic_acc:.3f} (improvement: {verification_improvement:+.3f})")
        print(f"  Correction rate: {correction_rate:.3f}")
        print(f"  False positive rate: {false_positive_rate:.3f}")
        print(f"  False negative rate: {false_negative_rate:.3f}")
        print(f"  Avg time (s): {agentic_avg_time:.2f}")

    if args.mode in ("majority", "both"):
        majority_acc = majority_correct_total / total
        majority_improvement = majority_acc - baseline_acc
        majority_avg_time = majority_time_total / total

        print(f"\n[Majority Voting @{args.votes}]")
        print(f"  Accuracy: {majority_acc:.3f} (improvement: {majority_improvement:+.3f})")
        print(f"  Avg time (s): {majority_avg_time:.2f}")
    print(f"Results written to: {args.output}")
    if args.summary_csv:
        print(f"Summary CSV written to: {args.summary_csv}")


if __name__ == "__main__":
    main()
