"""
agent_loop.py -- Structured Autonomous Browser Agent

Three-role architecture:
    OBSERVE  → REASONER  → PERCEIVER  → EXECUTE  → VERIFY  → BOOKKEEPING
    capture    text LLM     vision LLM   snap+click  vision LLM  memory

Sniper logic, vault system, memory compression, coordinate math: UNCHANGED.
"""

import cv2
import numpy as np
import json
import re
import time
import sys
import os

# ---------------------------------------------------------------
# Force unbuffered stdout so logs print in real-time
# ---------------------------------------------------------------
os.environ["PYTHONUNBUFFERED"] = "1"

from browser_engine import VisionBrowser
from ocr_sniper import find_text_and_click

# -- Lightweight subsystems --
from memory import create_memory, record_episode, compress_memory
from planner import create_plan
from policy import reason, perceive, verify, is_step_complete, call_llm


def _print(msg):
    """Print with immediate flush so logs appear in real-time in CMD."""
    print(msg, flush=True)


# ---------------------------------------------------------------
# 1. The OpenCV Sniper (IDENTICAL to V1)
# ---------------------------------------------------------------

def snap_to_element(image_bytes, qwen_x_norm, qwen_y_norm, search_radius=60):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]

    rough_x = int((qwen_x_norm / 1000.0) * width)
    rough_y = int((qwen_y_norm / 1000.0) * height)

    x_min, y_min = max(0, rough_x - search_radius), max(0, rough_y - search_radius)
    x_max, y_max = min(width, rough_x + search_radius), min(height, rough_y + search_radius)
    roi = img[y_min:y_max, x_min:x_max]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return rough_x, rough_y
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        local_cx, local_cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(largest_contour)
        local_cx, local_cy = x + w // 2, y + h // 2

    return x_min + local_cx, y_min + local_cy


# ---------------------------------------------------------------
# 2. Coordinate Extraction (handles every known hallucination)
# ---------------------------------------------------------------

def _extract_coords(target_str):
    """Extract coordinates from any format the model might produce."""
    if not target_str:
        return None, None

    # 1. Official Qwen box format
    match = re.search(r'<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>', target_str)
    # 2. Raw array format [x1,y1,x2,y2]
    if not match: match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', target_str)
    # 3. box_end typo
    if not match: match = re.search(r'<\|box_end\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>', target_str)
    # 4. box_start typo
    if not match: match = re.search(r'<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_start\|>', target_str)
    # 5. C++ hallucination
    if not match: match = re.search(r'CRect\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
    # 6. Apple iOS hallucination
    if not match: match = re.search(r'CGRect\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
    # 7. Tool-calling hallucination
    if not match: match = re.search(r'<tool_response>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)[\s\n]*\(\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
    # 8. Bare (x1,y1),(x2,y2) with any prefix
    if not match: match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)

    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1 + x2) // 2, (y1 + y2) // 2

    # 9. Single-point: <|box_start|>(x,y)<|box_end|> or bare (x,y)
    single = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
    if single:
        x, y = map(int, single.groups())
        return x, y

    return None, None


# ---------------------------------------------------------------
# 3. Vault Data Resolver
# ---------------------------------------------------------------

def _resolve_vault_text(raw_text):
    if not raw_text.startswith("$"):
        return raw_text, raw_text
    key = raw_text[1:]
    try:
        with open("vault.json", "r") as f:
            data_vault = json.load(f)
        if key in data_vault:
            _print(f"  [VAULT] Injecting secure data for '{key}'")
            return data_vault[key], f"Secure Data: {key}"
        else:
            _print(f"  [WARN] Missing vault key '{key}'.")
    except Exception:
        pass
    return raw_text, raw_text


# ---------------------------------------------------------------
# 4. The Autonomous Loop (Reasoner → Perceiver → Verifier)
# ---------------------------------------------------------------

def run_autonomous_agent(start_url, goal, max_steps=10):
    agent = VisionBrowser(headless=False)
    agent.navigate(start_url)

    _print(f"\n========================================")
    _print(f" AGENT DEPLOYED: '{goal}'")
    _print(f"========================================\n")

    # -- Structured memory --
    memory = create_memory(goal)
    action_history = []

    # -- Load vault keys --
    try:
        with open("vault.json", "r") as f:
            vault_keys = list(json.load(f).keys())
    except FileNotFoundError:
        vault_keys = []

    # -- Generate plan --
    _print("[Agent] Generating plan ...")
    memory["plan"] = create_plan(goal, call_llm)
    _print("")

    step = 1
    fail_count = 0  # Consecutive fails on the current step
    MAX_RETRIES = 3

    while step <= max_steps:
        _print(f"\n--- Step {step} ---")
        time.sleep(1.0)

        # ========================================
        # 1. OBSERVE (before screenshot)
        # ========================================
        _print("  [OBSERVE] Capturing screen ...")
        before_b64, cv_bytes = agent.capture_vision_state()

        # ========================================
        # 2. REASONER (text-only, no image)
        # ========================================
        current_plan_step = ""
        if memory["plan"] and memory["current_step"] < len(memory["plan"]):
            current_plan_step = memory["plan"][memory["current_step"]]
        _print(f"  [PLAN] Step: \"{current_plan_step}\"")

        _print("  [REASONER] Deciding what to do ...")
        current_url = agent.page.url
        _print(f"  [URL] {current_url}")
        decision = reason(
            goal=goal,
            plan_step=current_plan_step,
            summary=memory["summary"],
            action_history=action_history,
            vault_keys=vault_keys,
            current_url=current_url,
        )

        action_type = decision.get("action")
        element_desc = decision.get("element_description", "")
        expected_change = decision.get("expected_change", "")

        # -- Handle "done" immediately (no perceiver/verifier needed) --
        if action_type == "done":
            done_summary = decision.get("summary", "No summary provided.")
            _print(f"\n  MISSION ACCOMPLISHED!")
            _print(f"  ========================================")
            _print(f"  {done_summary}")
            _print(f"  ========================================")
            record_episode(memory, step, decision, "done")
            break

        # -- Handle reasoner errors --
        if action_type == "error":
            _print(f"  [ERROR] Reasoner: {decision.get('message', 'unknown')}")
            action_history.append({"action": "error", "status": "FAILED"})
            record_episode(memory, step, decision, "failure")
            fail_count += 1
            if fail_count >= MAX_RETRIES:
                _print(f"  [STUCK] {MAX_RETRIES} consecutive fails. Stopping.")
                break
            step += 1
            continue

        _print(f"  [REASONER] Action: {action_type}")
        _print(f"  [REASONER] Element: {element_desc}")
        _print(f"  [REASONER] Expects: {expected_change}")

        # ========================================
        # 3. PERCEIVER (vision, coordinates only)
        # ========================================
        _print("  [PERCEIVER] Locating element ...")
        coord_str = perceive(element_desc, before_b64)

        if "NOT_FOUND" in coord_str.upper():
            _print("  [PERCEIVER] Element NOT_FOUND.")
            action_history.append({
                "action": action_type,
                "element": element_desc,
                "status": "FAILED",
                "reason": "Perceiver could not find element"
            })
            record_episode(memory, step, decision, "failure")
            fail_count += 1
            if fail_count >= MAX_RETRIES:
                _print(f"  [STUCK] {MAX_RETRIES} consecutive fails. Stopping.")
                break
            step += 1
            continue

        rough_x, rough_y = _extract_coords(coord_str)
        if rough_x is None or rough_y is None:
            _print(f"  [PERCEIVER] Could not parse coordinates from: {coord_str}")
            action_history.append({
                "action": action_type,
                "element": element_desc,
                "status": "FAILED",
                "reason": "Bad coordinate format"
            })
            record_episode(memory, step, decision, "failure")
            fail_count += 1
            if fail_count >= MAX_RETRIES:
                _print(f"  [STUCK] {MAX_RETRIES} consecutive fails. Stopping.")
                break
            step += 1
            continue

        # ========================================
        # 4. EXECUTE (snap_to_element → click/type)
        # ========================================
        if action_type == "click":
            exact_x, exact_y = snap_to_element(cv_bytes, rough_x, rough_y)
            _print(f"  [EXECUTE] Click at ({exact_x}, {exact_y})")
            agent.click(exact_x, exact_y)

        elif action_type == "type":
            raw_text = decision.get("text", "")
            text_to_type, memory_log = _resolve_vault_text(raw_text)
            exact_x, exact_y = snap_to_element(cv_bytes, rough_x, rough_y)
            _print(f"  [EXECUTE] Focus ({exact_x}, {exact_y}), type: \"{memory_log}\"")
            agent.click(exact_x, exact_y)
            agent.type_text(text_to_type)

        else:
            _print(f"  [ERROR] Unknown action: {action_type}")
            action_history.append({"action": str(action_type), "status": "FAILED"})
            record_episode(memory, step, decision, "failure")
            fail_count += 1
            if fail_count >= MAX_RETRIES:
                _print(f"  [STUCK] {MAX_RETRIES} consecutive fails. Stopping.")
                break
            step += 1
            continue

        # ========================================
        # 5. OBSERVE AFTER + VERIFY
        # ========================================
        # type actions press Enter → page navigates → need longer settle time
        # click actions just interact with the DOM → short settle is fine
        if action_type == "type":
            _print("  [VERIFY] Waiting for page to settle after Enter ...")
            time.sleep(2.5)
        else:
            time.sleep(1.0)

        _print("  [VERIFY] Capturing after-state ...")
        after_b64, _ = agent.capture_vision_state()

        _print(f"  [VERIFY] Checking: \"{expected_change}\" ...")
        result = verify(expected_change, before_b64, after_b64)

        # ========================================
        # 6. BOOKKEEPING
        # ========================================
        if result == "SUCCESS":
            _print(f"  [RESULT] SUCCESS")
            history_entry = {"action": action_type, "element": element_desc, "status": "completed"}
            if action_type == "type":
                history_entry["text"] = decision.get("text", "")
            action_history.append(history_entry)
            record_episode(memory, step, decision, "success")
            fail_count = 0  # Reset on success

            # Check if the plan step is complete based on world state
            if memory["current_step"] < len(memory["plan"]):
                step_done = is_step_complete(
                    plan_step=current_plan_step,
                    current_url=agent.page.url,
                    action_history=action_history,
                )
                if step_done:
                    memory["current_step"] += 1
                    if memory["current_step"] < len(memory["plan"]):
                        _print(f"  [PLAN] Step complete. Advanced to: \"{memory['plan'][memory['current_step']]}\"")
                    else:
                        _print(f"  [PLAN] All plan steps complete.")

        else:
            _print(f"  [RESULT] FAIL — action did not have expected effect")
            action_history.append({
                "action": action_type,
                "element": element_desc,
                "status": "FAILED",
                "reason": "Verifier said FAIL"
            })
            record_episode(memory, step, decision, "failure")
            fail_count += 1
            if fail_count >= MAX_RETRIES:
                _print(f"  [STUCK] {MAX_RETRIES} consecutive fails on step. Stopping.")
                break

        # Compress memory every 5 steps
        compress_memory(memory, call_llm, interval=5)

        step += 1

    agent.page.wait_for_timeout(5000)
    agent.close()


# ---------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------

if __name__ == "__main__":
    test_goal = "Go to pewdiepie's channel and play his latest video. If you see an AD playing, wait for sometime till the skip button appears and click it."
    run_autonomous_agent("https://www.youtube.com", test_goal, max_steps=15)