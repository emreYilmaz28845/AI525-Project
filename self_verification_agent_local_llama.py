#!/usr/bin/env python
# coding: utf-8

"""
Self-verification agent using two local Ollama LLMs.

Prerequisites:
- Install Ollama: https://ollama.com/download
- Pull models, e.g.:
  ollama pull llama3.2:3b
  ollama pull qwen2.5:7b

Usage:
  python self_verification_agent_local_llama.py --question "Explain X"
  python self_verification_agent_local_llama.py --interactive
"""

import argparse
import json
import os
import re
import time
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


SOLVER_MODEL = os.getenv("SOLVER_MODEL", "llama3.2:3b")
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "qwen2.5:7b")

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))
MAX_VERIFIER_RETRIES = int(os.getenv("MAX_VERIFIER_RETRIES", "2"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "11"))
ANSWER_TOLERANCE = float(os.getenv("ANSWER_TOLERANCE", "0.01"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
LOG_PATH = os.getenv("VERIFICATION_LOG", "verification_log.jsonl")


SOLVER_SYSTEM = SystemMessage(
    content=(
        "You are a mathematical problem solver.\n\n"
        "When solving initially:\n"
        "- Show clear step-by-step reasoning\n"
        "- End with: Final Answer: <number>\n\n"
        "When receiving verification feedback:\n"
        "- Read the verifier's solution carefully\n"
        "- Compare it to your approach\n"
        "- If the verifier found an error, acknowledge it and fix it\n"
        "- If you still believe your answer is correct, explain why\n"
        "- Show all arithmetic clearly"
    )
)

VERIFIER_SYSTEM = SystemMessage(
    content=(
        "You are a mathematical verification specialist. Your job is to catch errors.\n\n"
        "PROCESS:\n"
        "1. Solve the problem independently step-by-step\n"
        "2. Extract the assistant's final numeric answer\n"
        "3. Compare both solutions carefully\n"
        "4. Check your arithmetic twice\n\n"
        "DECISION RULES:\n"
        "- If answers match within 1% tolerance → ACCEPT\n"
        "- If you found a specific calculation error → REVISE\n"
        "- If methods differ but both seem valid → Show your work, then decide\n\n"
        "FORMAT:\n"
        "My Solution: [your step-by-step work]\n"
        "Assistant's Answer: [extracted number]\n"
        "My Answer: [your number]\n"
        "Comparison: [detailed comparison]\n"
        "DECISION: [ACCEPT or REVISE]"
    )
)


def build_llms() -> tuple[ChatOllama, ChatOllama]:
    common_kwargs = {"request_timeout": REQUEST_TIMEOUT} if REQUEST_TIMEOUT else {}
    solver = ChatOllama(model=SOLVER_MODEL, temperature=0.3, **common_kwargs)
    verifier = ChatOllama(model=VERIFIER_MODEL, temperature=0.0, **common_kwargs)
    return solver, verifier


def extract_confidence(text: str) -> Optional[float]:
    if not text:
        return None
    label_match = re.search(
        r"confidence[:\s]*([0-9]+(?:\.\d+)?)(?:\s*/\s*10)?", text, flags=re.IGNORECASE
    )
    if label_match:
        try:
            return max(0.0, min(10.0, float(label_match.group(1))))
        except ValueError:
            return None
    # Fallback: grab the last small integer at the end if it looks like a rating
    tail_numbers = re.findall(r"\b(10|[0-9])(?:\.\d+)?\b", text[-20:])
    if tail_numbers:
        try:
            return max(0.0, min(10.0, float(tail_numbers[-1])))
        except ValueError:
            return None
    return None


def strip_confidence(text: str) -> tuple[str, Optional[float]]:
    confidence = extract_confidence(text)
    cleaned = re.sub(
        r"(?im)^.*confidence[:\s]*[0-9]+(?:\.\d+)?(?:\s*/\s*10)?.*$", "", text
    )
    cleaned = re.sub(
        r"\(?\s*confidence[:\s]*[0-9]+(?:\.\d+)?(?:\s*/\s*10)?\s*\)?", "", cleaned, flags=re.IGNORECASE
    )
    cleaned = cleaned.strip()
    return cleaned or text, confidence


def extract_final_number(text: str) -> Optional[float]:
    """Extract the final numeric answer from text."""
    if not text:
        return None
    final_match = re.search(
        r"final answer[:\s]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if final_match:
        try:
            return float(final_match.group(1))
        except ValueError:
            pass

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def parse_decision(text: str) -> str:
    if not text:
        return "REVISE"
    for line in text.splitlines()[::-1]:
        normalized = line.strip().upper()
        if "DECISION:" in normalized:
            after = normalized.split("DECISION:", 1)[-1].strip()
            if "ACCEPT" in after:
                return "ACCEPT"
            if "REVISE" in after:
                return "REVISE"
        if normalized.startswith("DECISION:"):
            return "ACCEPT" if "ACCEPT" in normalized else "REVISE"
    return "REVISE"


def has_critique_and_suggestions(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    has_critique = "critique" in lowered
    has_suggestions = "suggest" in lowered
    return has_critique and has_suggestions


def answers_are_equivalent(answer1: str, answer2: str, tolerance: float = ANSWER_TOLERANCE) -> bool:
    numbers1 = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", answer1 or "")
    numbers2 = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", answer2 or "")
    if not numbers1 or not numbers2:
        return False
    try:
        val1 = float(numbers1[-1])
        val2 = float(numbers2[-1])
    except ValueError:
        return False
    if val2 == 0:
        return abs(val1 - val2) < tolerance
    return abs(val1 - val2) / abs(val2) < tolerance


def is_valid_critique(critique_text: str, original_answer: str) -> bool:
    if not critique_text:
        return False
    if len(critique_text) < 50:
        return False
    reasoning_keywords = [
        "calculate",
        "error",
        "incorrect",
        "mistake",
        "should be",
        "actually",
    ]
    has_reasoning = any(kw in critique_text.lower() for kw in reasoning_keywords)
    has_specifics = bool(re.search(r"\d", critique_text))
    mentions_answer = answers_are_equivalent(critique_text, original_answer) or bool(
        re.search(r"final answer", critique_text, flags=re.IGNORECASE)
    )
    return has_reasoning and has_specifics and mentions_answer


def build_feedback_message(critique_text: str) -> HumanMessage:
    return HumanMessage(
        content=(
            "Verifier feedback about your last answer:\n"
            f"{critique_text}\n"
            "Revise only to fix arithmetic/units or clear logical errors. "
            "Keep the answer concise and end with 'Final Answer: <number>'."
        )
    )


def enforce_verifier_structure(
    verifier: ChatOllama, messages: List, critique: AIMessage
) -> AIMessage:
    content = (critique.content or "").strip()
    if has_critique_and_suggestions(content):
        return critique

    for attempt in range(1, MAX_VERIFIER_RETRIES + 1):
        retry_msgs = messages + [
            HumanMessage(
                content=(
                    "Your response must include:\n"
                    "1) A critique section labeled 'Critique:'\n"
                    "2) A suggestions section labeled 'Suggestions:' with 1-3 items\n"
                    "3) A final line 'DECISION: ACCEPT' or 'DECISION: REVISE'\n"
                    "Please respond again with that exact structure."
                )
            )
        ]
        critique = verifier.invoke(retry_msgs)
        content = (critique.content or "").strip()
        if has_critique_and_suggestions(content):
            print(f"[Verifier] Structured output recovered on retry {attempt}.")
            return critique

    fallback = (
        "Critique: Verifier did not provide a critique.\n"
        "Suggestions: Please provide a critique and 1-3 suggestions.\n"
        "DECISION: REVISE"
    )
    print("[Verifier] Missing critique/suggestions; using fallback critique.")
    return AIMessage(content=fallback)


def call_verifier_with_retries(
    verifier: ChatOllama, messages: List
) -> AIMessage:
    critique = verifier.invoke(messages)
    content = (critique.content or "").strip()
    if content:
        return critique

    for attempt in range(1, MAX_VERIFIER_RETRIES + 1):
        retry_msgs = messages + [
            HumanMessage(
                content=(
                    "Your last response was empty. Provide a critique and end with "
                    "DECISION: ACCEPT or DECISION: REVISE."
                )
            )
        ]
        critique = verifier.invoke(retry_msgs)
        content = (critique.content or "").strip()
        if content:
            print(f"[Verifier] Empty output; recovered on retry {attempt}.")
            return critique

    fallback = (
        "Critique: Verifier returned empty output after retries.\n"
        "Suggestions: Provide a critique and 1-3 suggestions.\n"
        "DECISION: REVISE"
    )
    print("[Verifier] Empty output after retries; using fallback critique.")
    return AIMessage(content=fallback)


def log_attempt(
    attempt_num: int,
    solver_answer: str,
    verifier_critique: str,
    decision: str,
    solver_num: Optional[float],
    verifier_num: Optional[float],
) -> None:
    if not LOG_PATH:
        return
    entry = {
        "attempt": attempt_num,
        "solver_answer": solver_answer,
        "solver_number": solver_num,
        "verifier_critique": verifier_critique,
        "verifier_number": verifier_num,
        "decision": decision,
        "timestamp": time.time(),
    }
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception:
        # Logging should never break the run.
        pass


def run_self_verification(
    solver: ChatOllama, verifier: ChatOllama, question: str
) -> str:
    initial_prompt = (
        f"{question}\n\n"
        "After your answer, add a separate line 'Confidence: <0-10>' to rate your certainty."
    )
    history: List = [HumanMessage(content=initial_prompt)]
    last_solver_answer: Optional[str] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        solver_msgs = [SOLVER_SYSTEM] + history
        solver_resp = solver.invoke(solver_msgs)
        cleaned_answer, solver_confidence = strip_confidence(solver_resp.content or "")
        last_solver_answer = cleaned_answer
        history.append(AIMessage(content=cleaned_answer))
        print(f"\n[Solver] Attempt {attempt} Answer:")
        print(cleaned_answer)

        if attempt == 1 and solver_confidence is not None:
            print(f"[System] Solver reported confidence {solver_confidence}/10.")

        verifier_msgs = [VERIFIER_SYSTEM] + history
        critique = call_verifier_with_retries(verifier, verifier_msgs)
        critique = enforce_verifier_structure(verifier, verifier_msgs, critique)
        critique_text = (critique.content or "").strip()

        print("=== RAW VERIFIER OUTPUT ===")
        print(critique_text)
        print("===========================")

        decision = parse_decision(critique_text)
        history.append(critique)

        solver_num = extract_final_number(last_solver_answer or "")
        verifier_num = extract_final_number(critique_text or "")
        log_attempt(attempt, last_solver_answer or "", critique_text, decision, solver_num, verifier_num)

        if decision == "ACCEPT":
            print("[Verifier] DECISION: ACCEPT")
            break

        if attempt >= MAX_ATTEMPTS:
            print(f"[Verifier] Max attempts reached ({attempt}), forcing ACCEPT.")
            break

        history.append(build_feedback_message(critique_text))
        print("[Verifier] DECISION: REVISE")

    final_answer = last_solver_answer or ""

    if not re.search(r"\d", final_answer or ""):
        try:
            follow_up = solver.invoke(
                [
                    SOLVER_SYSTEM,
                    HumanMessage(
                        content=(
                            "From your prior reasoning, provide ONLY the final numeric "
                            "answer. If no numeric answer exists, reply: No numeric answer."
                        )
                    ),
                    AIMessage(content=final_answer),
                ]
            )
            extracted = (follow_up.content or "").strip()
            if extracted:
                final_answer = f"{final_answer}\nFinal Answer Extracted: {extracted}"
        except Exception:
            pass

    return final_answer


def run_baseline(solver: ChatOllama, question: str) -> str:
    msgs = [SOLVER_SYSTEM, HumanMessage(content=question)]
    return solver.invoke(msgs).content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="Question to answer.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for a question in the terminal.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also print a single-shot baseline answer.",
    )
    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.error("Provide --question or use --interactive.")

    solver, verifier = build_llms()

    def answer_once(q: str) -> None:
        if args.baseline:
            base_answer = run_baseline(solver, q)
            print(f"[Baseline] Answer: {base_answer}")
        final_answer = run_self_verification(solver, verifier, q)
        print("\n[Self-Verification] Final Answer:")
        print(final_answer)

    if args.interactive:
        print("Interactive mode. Type 'exit' or press Enter on empty input to quit.")
        while True:
            question = input("Enter a question: ").strip()
            if not question or question.lower() in {"exit", "quit"}:
                break
            answer_once(question)
    else:
        question = args.question
        answer_once(question)


if __name__ == "__main__":
    main()
