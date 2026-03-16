"""
planner.py — Goal Decomposition via LLM

Calls the LLM once before the agent loop to decompose a high-level goal
into an ordered list of concrete browser subtasks.

Public API:
    create_plan(goal, llm_fn) -> list[str]
"""

import json
import re


def create_plan(goal, llm_fn):
    """
    Decompose a user goal into 3-8 ordered browser subtasks.

    Args:
        goal:   The user's natural-language goal string.
        llm_fn: Callable(prompt, system_prompt) -> str.  Text-only LLM call.

    Returns:
        A list of step strings, e.g.:
        ["open youtube", "search for pewdiepie", "click subscribe"]
    """
    system_prompt = (
        "You are a web task planner. Given a user goal, decompose it into "
        "3-8 concrete, ordered browser actions. Each step should be a short "
        "imperative sentence describing one browser interaction.\n\n"
        "Output ONLY a JSON array of strings. No extra text."
    )

    prompt = (
        f"User goal: \"{goal}\"\n\n"
        f"Decompose this into ordered browser steps. "
        f"Output a JSON array of strings."
    )

    try:
        raw = llm_fn(prompt, system_prompt)

        # Extract JSON array from response (tolerant of markdown fences)
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group(0))
            if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
                print(f"[Planner] Generated {len(plan)}-step plan:")
                for i, step in enumerate(plan):
                    print(f"  {i+1}. {step}")
                return plan

        # Fallback: try line-by-line parsing
        lines = [l.strip().lstrip("0123456789.-) ") for l in raw.strip().splitlines() if l.strip()]
        lines = [l for l in lines if len(l) > 3]
        if lines:
            print(f"[Planner] Parsed {len(lines)}-step plan (fallback):")
            for i, step in enumerate(lines):
                print(f"  {i+1}. {step}")
            return lines

    except Exception as e:
        print(f"[Planner] Plan generation failed: {e}")

    # Ultimate fallback: single-step plan
    fallback = [goal]
    print(f"[Planner] Using single-step fallback plan: {fallback}")
    return fallback
