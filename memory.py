"""
memory.py — Structured Agent Memory

Manages the agent's episodic memory, working state, and compressed summaries.
Replaces the flat `action_history` list with a rich memory object.

Public API:
    create_memory(goal)                              -> dict
    record_episode(memory, step, action, result, prev_state) -> None
    compress_memory(memory, llm_fn, interval=5)      -> None
    advance_plan(memory, llm_fn, current_state)      -> bool
    detect_failure(prev_state, new_state, action)     -> str
"""

import json
import copy


# ──────────────────────────────────────────────
# Memory Initialisation
# ──────────────────────────────────────────────

def create_memory(goal):
    """
    Build the initial structured memory object.

    Args:
        goal: The original user goal string.

    Returns:
        dict with keys: goal, plan, current_step, working_state, episodes, summary.
    """
    return {
        "goal": goal,
        "plan": [],              # Filled by planner before the loop starts
        "current_step": 0,       # Index into plan
        "working_state": {},     # Latest page state from perception
        "episodes": [],          # Recent step records
        "summary": "",           # Compressed summary of older episodes
    }


# ──────────────────────────────────────────────
# Episode Recording
# ──────────────────────────────────────────────

def record_episode(memory, step, action, result, prev_state=None):
    """
    Append a structured episode after every action.

    Args:
        memory:     The structured memory dict.
        step:       Current loop step number (int).
        action:     The action dict returned by the policy (e.g. {"action": "click", "element_id": 4}).
        result:     Outcome string — "success", "failure", or descriptive text.
        prev_state: Optional previous working_state snapshot for comparison.
    """
    plan_step = ""
    if memory["plan"] and memory["current_step"] < len(memory["plan"]):
        plan_step = memory["plan"][memory["current_step"]]

    episode = {
        "step": step,
        "plan_step": plan_step,
        "state_summary": _summarize_state(memory["working_state"]),
        "action": action,
        "result": result,
    }
    memory["episodes"].append(episode)


def _summarize_state(state):
    """Produce a compact text summary of the working state for episode storage."""
    if not state:
        return "no state captured"
    page_type = state.get("page_type", "unknown")
    n_buttons = len(state.get("buttons", []))
    n_text = len(state.get("text_elements", []))
    n_search = len(state.get("search_boxes", []))
    return f"page={page_type}, buttons={n_buttons}, text={n_text}, search_boxes={n_search}"


# ──────────────────────────────────────────────
# Memory Compression
# ──────────────────────────────────────────────

def compress_memory(memory, llm_fn, interval=5):
    """
    Every `interval` episodes, summarize older ones via the LLM and truncate.

    Args:
        memory:   The structured memory dict.
        llm_fn:   Callable(prompt, system_prompt) -> str.  LLM text generation.
        interval: How many episodes to accumulate before compressing.
    """
    if len(memory["episodes"]) < interval:
        return  # Not enough episodes to compress yet

    # Keep the most recent `interval // 2` episodes intact for immediate context
    keep = max(interval // 2, 2)
    old_episodes = memory["episodes"][:-keep]
    recent_episodes = memory["episodes"][-keep:]

    # Build a prompt for the LLM
    episodes_text = json.dumps(old_episodes, indent=2, default=str)
    existing_summary = memory["summary"] or "None yet."

    prompt = (
        f"You are a memory compression agent.\n"
        f"Existing summary of earlier activity:\n{existing_summary}\n\n"
        f"New episodes to incorporate:\n{episodes_text}\n\n"
        f"Produce a concise summary (max 200 words) that captures all important "
        f"facts, outcomes, and progress. Include: what pages were visited, what "
        f"actions succeeded or failed, and what the agent accomplished so far."
    )

    try:
        compressed = llm_fn(prompt, "You are a concise summarizer. Output only the summary text.")
        memory["summary"] = compressed.strip()
    except Exception as e:
        # Non-fatal: keep the older summary if LLM call fails
        print(f"[Memory] Compression failed: {e}")

    # Truncate to only recent episodes
    memory["episodes"] = recent_episodes


# ──────────────────────────────────────────────
# Plan Progress Tracking
# ──────────────────────────────────────────────

def advance_plan(memory, llm_fn, current_state):
    """
    Ask the LLM whether the current plan step has been completed.
    If yes, increment current_step.

    Args:
        memory:        The structured memory dict.
        llm_fn:        Callable(prompt, system_prompt) -> str.
        current_state: The current page state dict from perception.

    Returns:
        True if the plan step was advanced, False otherwise.
    """
    if not memory["plan"]:
        return False
    if memory["current_step"] >= len(memory["plan"]):
        return False  # All steps done

    current_plan_step = memory["plan"][memory["current_step"]]
    state_text = json.dumps(current_state, indent=2, default=str)

    # Summarize recent episodes for context
    recent = memory["episodes"][-3:] if memory["episodes"] else []
    recent_text = json.dumps(recent, indent=2, default=str)

    prompt = (
        f"Current plan step: \"{current_plan_step}\"\n\n"
        f"Current page state:\n{state_text}\n\n"
        f"Recent actions:\n{recent_text}\n\n"
        f"Has this plan step been completed? Answer ONLY 'yes' or 'no'."
    )

    try:
        answer = llm_fn(prompt, "You are a progress checker. Answer ONLY 'yes' or 'no'.")
        if "yes" in answer.strip().lower():
            memory["current_step"] += 1
            print(f"[Memory] ✅ Plan step completed: \"{current_plan_step}\" → advancing to step {memory['current_step']}")
            return True
    except Exception as e:
        print(f"[Memory] Plan advance check failed: {e}")

    return False


# ──────────────────────────────────────────────
# Failure Detection
# ──────────────────────────────────────────────

def detect_failure(prev_state, new_state, action):
    """
    Compare page states before and after an action to detect failures.

    Returns:
        "success"            — if page state changed meaningfully.
        "no_change"          — if the page looks identical (nothing happened).
        "element_missing"    — if the targeted element type is absent on the new page.
    """
    if not prev_state or not new_state:
        return "success"  # Can't compare — assume OK

    # Check if page type changed (navigation happened)
    if prev_state.get("page_type") != new_state.get("page_type"):
        return "success"

    # Check if visible element count changed meaningfully (±3 elements)
    prev_count = len(prev_state.get("visible_elements", []))
    new_count = len(new_state.get("visible_elements", []))
    if abs(prev_count - new_count) > 2:
        return "success"

    # Check if the text content changed
    prev_texts = set(e.get("text", "") for e in prev_state.get("visible_elements", []))
    new_texts = set(e.get("text", "") for e in new_state.get("visible_elements", []))
    if len(prev_texts.symmetric_difference(new_texts)) > 2:
        return "success"

    # Nothing meaningful changed
    return "no_change"
