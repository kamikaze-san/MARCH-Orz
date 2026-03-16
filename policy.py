"""
policy.py -- Three-Role LLM System: Reasoner, Perceiver, Verifier

Reasoner:  Text-only. Decides WHAT to interact with next and WHY.
Perceiver: Vision. Finds WHERE that element is on screen (coordinates only).
Verifier:  Vision. Checks if the action had the expected effect (SUCCESS/FAIL).

Public API:
    reason(goal, plan_step, summary, action_history, vault_keys) -> dict
    perceive(element_description, base64_image) -> str
    verify(expected_change, before_b64, after_b64) -> str
    call_llm(prompt, system_prompt, base64_image=None) -> str
"""

import ollama
import json
import re

# ---------------------------------------------------------------
# LLM Interface (shared by all modules)
# ---------------------------------------------------------------

_MODEL = "qwen3-vl:8b-instruct"


def call_llm(prompt, system_prompt, base64_image=None):
    """
    Thin wrapper around ollama.chat(). Used by planner, memory,
    and internally by reason/perceive/verify.
    """
    user_msg = {"role": "user", "content": prompt}
    if base64_image:
        user_msg["images"] = [base64_image]

    response = ollama.chat(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            user_msg,
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------
# 1. REASONER — Text-only, decides WHAT to interact with
# ---------------------------------------------------------------

def reason(goal, plan_step, summary, action_history, vault_keys, current_url=""):
    """
    Text-only LLM call. No image. Decides the next action and describes
    the target element for the Perceiver to locate.

    Returns dict:
        {"action": "click"|"type"|"done",
         "element_description": "the YouTube search bar at the top center",
         "expected_change": "search results page loads with video grid",
         "text": "pewdiepie"}   # only for type actions
    """
    history_str = json.dumps(action_history, indent=2) if action_history else "[]"

    system_prompt = (
        "You are a task reasoner. You never see screenshots. "
        "Given a goal, current URL, and action history, output only: what specific UI element "
        "to look for next, and what change you expect after interacting with it. "
        "Be specific about visual appearance and screen location."
    )

    parts = []
    parts.append(f"Goal: {goal}")
    if current_url:
        parts.append(f"Current Page URL: {current_url}")
    if plan_step:
        parts.append(f"Current Plan Step: {plan_step}")
    if summary:
        parts.append(f"Memory Summary: {summary}")
    parts.append(f"Available Secure Data Keys: {vault_keys}")
    parts.append(f"Past Actions (DO NOT REPEAT FAILED ONES):\n{history_str}")
    parts.append("""Decide the SINGLE next action. Output strict JSON with these keys:

{"action": "click", "element_description": "describe the exact element to find on screen", "expected_change": "significant UI change visible after clicking (e.g. new page loads, modal opens, button changes state)"}
{"action": "type", "element_description": "describe the text input field", "expected_change": "significant UI change visible AFTER Enter is pressed (e.g. search results page appears, form submits, new content loads)", "text": "exact text to type or $vault_key"}
{"action": "done", "summary": "detailed description of what was accomplished"}

RULES:
1. "element_description" must describe visual appearance and location (e.g. "the search bar at the top center of the page").
2. TO ENTER TEXT: NEVER use "click" on a text field or search bar. Use "type" directly. The system clicks the field, types the text, and presses Enter automatically.
3. "expected_change" must describe a SIGNIFICANT DOM change: a new page, new section, new element, or a changed element. Not pixel-level changes like cursor blinking.
4. Read the Current Page URL carefully. It tells you what page you are on. Do not describe elements from a previous page.
5. If the last action FAILED, describe a DIFFERENT element or approach.
6. "done" means the ENTIRE goal is complete. Do NOT use "done" for sub-steps.
7. Complete ALL plan steps before using "done".""")

    prompt = "\n".join(parts)
    raw = call_llm(prompt, system_prompt)
    print(f"\n   [Reasoner]: {raw}", flush=True)

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {"action": "error", "message": "Reasoner failed to produce valid JSON."}


# ---------------------------------------------------------------
# 2. PERCEIVER — Vision, finds WHERE the element is (coords only)
# ---------------------------------------------------------------

def perceive(element_description, base64_image):
    """
    Vision LLM call. Receives a screenshot and a description of one
    specific element. Returns bounding box coordinates or "NOT_FOUND".

    Returns:
        Raw coordinate string like "<|box_start|>(x1,y1),(x2,y2)<|box_end|>"
        or "NOT_FOUND".
    """
    system_prompt = (
    "You are a visual element locator. You receive a screenshot and a description of one specific UI element. "
    "Your only job is to find that element in the image and output its bounding box. "
    "You MUST output coordinates in EXACTLY this format: য়(x1,y1),(x2,y2)য় "
    "where x and y are integers from 0 to 1000. "
    "If you can see the element, output the box. "
    "If you genuinely cannot find the element after looking carefully, output only: NOT_FOUND "
    "Do not output anything else."
)

    prompt = (
        f"Find this element in the screenshot: {element_description}\n"
        f"Look carefully at the full image before responding."
    )
    raw = call_llm(prompt, system_prompt, base64_image=base64_image)
    print(f"   [Perceiver]: {raw}", flush=True)

    return raw.strip()


# ---------------------------------------------------------------
# 3. VERIFIER — Vision, checks if action had expected effect
# ---------------------------------------------------------------

def verify(expected_change, before_b64, after_b64):
    """
    Vision LLM call with TWO images. Checks if the expected change
    is visible in the after screenshot.

    Returns: "SUCCESS" or "FAIL"
    """
    system_prompt = (
        "You are a verification checker. You receive two screenshots and an "
        "expected change. Output ONLY: SUCCESS if the expected change is visible "
        "in the after screenshot, FAIL if it is not. Nothing else."
    )

    prompt = f"Expected change: {expected_change}"

    user_msg = {
        "role": "user",
        "content": prompt,
        "images": [before_b64, after_b64],
    }

    response = ollama.chat(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            user_msg,
        ],
        options={"temperature": 0.0},
    )

    raw = response["message"]["content"].strip()
    print(f"   [Verifier]: {raw}", flush=True)

    # Extract SUCCESS or FAIL from whatever the model says
    if "SUCCESS" in raw.upper():
        return "SUCCESS"
    return "FAIL"


# ---------------------------------------------------------------
# 4. STEP COMPLETION — Text-only, checks if plan step is done
# ---------------------------------------------------------------

def is_step_complete(plan_step, current_url, action_history):
    """
    Text-only LLM call. Checks whether the current plan step has been
    accomplished based on the URL and action history (world state).

    Returns: True if the step is complete, False otherwise.
    """
    history_str = json.dumps(action_history[-6:], indent=2) if action_history else "[]"

    system_prompt = (
        "You are a task progress checker. You never see screenshots. "
        "Given a plan step, the current page URL, and recent action history, "
        "determine if the plan step has been accomplished. "
        "Output ONLY: COMPLETE if the step is done, INCOMPLETE if not. Nothing else."
    )

    prompt = (
        f"Plan step: {plan_step}\n"
        f"Current URL: {current_url}\n"
        f"Recent actions:\n{history_str}"
    )

    raw = call_llm(prompt, system_prompt)
    print(f"   [StepCheck]: {raw}", flush=True)

    return "COMPLETE" in raw.upper()
