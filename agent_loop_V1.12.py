"""
agent_vision_only.py -- Pure Vision Browser Agent

Architecture:
    1. Vision Perception: Takes a compressed 512x512 JPEG of the viewport.
    2. State Awareness: Passes URL and Page Title to the VLM to anchor its completion logic.
    3. LLM Decision: qwen3-vl predicts coordinates or navigation actions.
    4. OpenCV Sniper: Edge detection snaps rough VLM coordinates to exact button centers.
    5. Playwright Execution: Performs the click/type/scroll/goto actions.
"""

import argparse
# import cv2  # leftover import, not currently used
import numpy as np
import ollama
import re
import json
import time
import base64
import io
import os
from PIL import Image
from playwright.sync_api import sync_playwright

os.environ["PYTHONUNBUFFERED"] = "1"

# --- Config ---
VISION_MODEL = "qwen3-vl:4b-instruct"  # default, overridable via --model
                                         # prefix with "hf:" for HuggingFace VL models
                                         # e.g. "hf:LiquidAI/LFM2.5-VL-1.6B"
PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".browser_profiles")

def _print(msg):
    print(msg, flush=True)


def _get_field(decision: dict, *keys):
    """Extract a field from top-level OR nested 'action-specific-fields'.
    Small models (4B) sometimes wrap fields inside that dict instead of top-level.
    Returns first non-None match, or None if not found anywhere.
    """
    for key in keys:
        val = decision.get(key)
        if val is not None:
            return val
    nested = decision.get("action-specific-fields", {})
    if isinstance(nested, dict):
        for key in keys:
            val = nested.get(key)
            if val is not None:
                return val
    return None


def _detect_loop(history: list, n: int = 3) -> str | None:
    """Return a warning string if the last n history entries are the same action.
    Checks action type + primary field (text / coords / url).
    Returns None if no loop detected.
    """
    if len(history) < n:
        return None
    last = history[-n:]
    actions = [h.get("action") for h in last]
    if len(set(actions)) != 1:
        return None   # different actions, no loop
    act = actions[0]
    if act == "type":
        vals = [h.get("text", "") for h in last]
    elif act == "click":
        vals = [h.get("coords", "") for h in last]
    elif act == "goto":
        vals = [h.get("url", "") for h in last]
    elif act == "scroll":
        vals = [h.get("direction", "down") for h in last]   # same direction n times IS a loop
    else:
        return None
    if len(set(vals)) == 1:
        return (
            f"⚠️ LOOP DETECTED: you have done '{act}' {n} times in a row "
            f"with no progress. Your last action did NOT work. "
            f"Do NOT repeat it. Try a completely different approach."
        )
    return None


# --- HuggingFace VL Backend ---
# Loaded lazily on first use, cached as singleton per model_id.
# Ollama is used when model name has no "hf:" prefix (default behaviour unchanged).

_hf_vl_cache: dict = {}

class HuggingFaceVLBackend:
    """Vision-Language backend using HuggingFace transformers.
    Requires: pip install transformers torch accelerate
    Tested with: LiquidAI/LFM2.5-VL-1.6B
    """
    def __init__(self, model_id: str):
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch
        except ImportError:
            raise ImportError("Run: pip install transformers torch accelerate")

        _print(f"[HF-VL] Loading '{model_id}' — first run will download weights ...")
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()
        device = next(self.model.parameters()).device
        _print(f"[HF-VL] Loaded on {device}")

    def generate(self, system_prompt: str, user_text: str, pil_image, max_new_tokens: int = 1024) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text": user_text},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,       # ignored when do_sample=False
                repetition_penalty=1.05,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0][input_len:]
        return self.processor.batch_decode([new_tokens], skip_special_tokens=True)[0]


def _get_hf_vl_backend(model_id: str) -> HuggingFaceVLBackend:
    """Return cached backend instance, loading on first call."""
    if model_id not in _hf_vl_cache:
        _hf_vl_cache[model_id] = HuggingFaceVLBackend(model_id)
    return _hf_vl_cache[model_id]

def safe_goto(page, url, timeout=45000):
    """Navigate without requiring noisy modern sites to reach networkidle."""
    page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    try:
        page.wait_for_load_state("networkidle", timeout=6000)
    except Exception:
        _print("  [NAV] Continuing before networkidle; page is still making background requests.")
    time.sleep(1.0)


def safe_page_state(page, retries=4):
    """Read URL + title safely, retrying if a redirect destroys the execution context mid-read.
    Amazon/modern SPAs often do a redirect chain after search, causing 'Execution context was
    destroyed' if we read too soon. We wait for domcontentloaded and retry up to `retries` times.
    """
    for attempt in range(retries):
        try:
            page.wait_for_load_state("domcontentloaded", timeout=15000)
            url   = page.url
            title = page.title()
            return url, title
        except Exception as e:
            if attempt == retries - 1:
                # Last attempt: return what we can, use placeholder for title
                try:
                    return page.url, "(loading)"
                except Exception:
                    return "about:blank", "(loading)"
            _print(f"  [STATE] Navigation in progress, retrying in 2s... ({e})")
            time.sleep(2.0)

# --- 1. Goal Refinement ---
def refine_goal(goal: str, model: str) -> str:
    """One-time pre-loop call: converts raw user goal into a short first-person
    exploration strategy. Text-only — no screenshot, no browser needed yet.
    Returns the strategy string, or empty string if the call fails.
    """
    _print(f"  [REFINE] Generating exploration strategy for goal...")
    system_prompt = (
        "You are preparing a browser agent to execute a web task. "
        "Given the user's goal, output a short exploration strategy in first-person format starting with 'I will'. "
        "Be concise and to the point — one short paragraph only. "
        "Write it as an exploratory direction, not step-by-step instructions. "
        "No UI element references, no numbered steps, no extra commentary."
    )
    try:
        if model.startswith("hf:"):
            # HF text-only path — use the VL backend but pass a blank 1x1 image
            from PIL import Image as PILImage
            blank = PILImage.new("RGB", (1, 1), color=(255, 255, 255))
            backend = _get_hf_vl_backend(model[3:])
            result = backend.generate(system_prompt, f"Goal: {goal}", blank, max_new_tokens=128)
        else:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Goal: {goal}"},
                ],
                options={"temperature": 0.0, "num_predict": 128},
            )
            result = response["message"]["content"].strip()

        # Strip any <think> blocks
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        _print(f"  [REFINE] Strategy: {result}\n")
        return result
    except Exception as e:
        _print(f"  [REFINE] Failed ({e}), continuing without strategy.")
        return ""


# --- 2. Screenshot Capture ---
def capture_screenshot(page, resize_dim=(1920, 1080)):
    """Capture full-res screenshot. Also returns a 2x zoomed crop of the top results area for detail."""
    raw_bytes = page.screenshot(type='jpeg', quality=100)
    image = Image.open(io.BytesIO(raw_bytes))

    # Full page at native res (for LLM context)
    full_buf = io.BytesIO()
    image.save(full_buf, format="JPEG", quality=90)
    full_b64 = base64.b64encode(full_buf.getvalue()).decode('utf-8')

    return full_b64, raw_bytes

# --- 2. OpenCV Sniper ---
def snap_to_element(image_bytes, qwen_x_norm, qwen_y_norm, search_radius=60):
    """Snaps rough normalized coordinates (0-1000) to exact button centers using edge detection."""
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
    
    if not contours: return rough_x, rough_y
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        local_cx, local_cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(largest_contour)
        local_cx, local_cy = x + w // 2, y + h // 2
        
    return x_min + local_cx, y_min + local_cy

# --- 3. Robust Coordinate Extractor ---
def _extract_coords(target_str):
    """Parses all known local VLM hallucination formats into X/Y centers."""
    if not target_str: return None, None
    
    match = re.search(r'<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*(?:<\|box_end\|>|<\|box_start\|>)', target_str)
    if not match: match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', target_str)
    if not match: match = re.search(r'(?:C|CG)Rect\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
    if not match: match = re.search(r'<tool_response>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)[\s\n]*\(\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
    
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1 + x2) // 2, (y1 + y2) // 2
        
    single = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
    if single:
        x, y = map(int, single.groups())
        return x, y
        
    return None, None

# --- 4. Vision Decision LLM ---
def get_vision_decision(base64_image, goal, history, current_url, page_title, tabs_str="[]", scratchpad=None, model=None, raw_image_bytes=None, warning=None, strategy=None):
    if scratchpad is None:
        scratchpad = {}
    history_str = json.dumps(history[-6:], indent=2) if history else "[]"
    scratchpad_str = json.dumps(scratchpad, indent=2) if scratchpad else "{}"

    strategy_line = f"\nSTRATEGY: {strategy}" if strategy else ""
    system_prompt = f"""You are an autonomous web agent. You navigate websites by looking at screenshots to complete goals.

GOAL: {goal}{strategy_line}

YOUR PROGRESS (trust this over what you see — it is your memory):
{scratchpad_str}

STATE:
- URL: {current_url}
- Title: {page_title}
- Tabs: {tabs_str}

RECENT ACTIONS:
{history_str}

Output a single JSON object with these fields in this exact order:
1. "think": read YOUR PROGRESS first, then reason through what is done, what is current, and what to do next
2. "observations": describe everything on screen relevant to the goal — be complete, do not skip items
3. "scratchpad": REQUIRED every step — update your progress tracker with these fields:
   - "plan": list of all subtasks to complete the goal (generate once on step 1, do not change after)
   - "done": list of subtasks you have fully completed so far
   - "current": the subtask you are doing right now
   - "next": the subtask after this one
   - "conclusion": set this when the goal is fully achieved, then output done
4. "action": one of goto / click / type / scroll / switch_tab / done
5. action-specific fields:
   - goto: "url"
   - click: "x","y" (0-1000 normalized to screen width/height)
   - type: "x","y" (0-1000, center of the input field) AND "text" (what to type) — BOTH are required
   - scroll: "direction" ("up" or "down")
   - switch_tab: "tab_index"

Scratchpad rules:
- Write scratchpad on EVERY step — it is not optional
- Only update fields that changed — other fields carry over automatically
- "done" grows as you complete subtasks — never shrink it
- If your last action had no visible effect, do NOT repeat it — try a different approach

Constraints:
- Cannot click the browser address bar — use goto to navigate to a URL
- Use type (not click) for any text input field — the system handles focusing
- When conclusion is set, output done — do not keep acting

Reading your action history:
- Each click logs a "changes" object showing what changed: url, title, tabs count
- Combine "changes" with the current screenshot to figure out what your last action did
- "tabs" went up → a new tab opened, use switch_tab(index) to go to it
- Nothing changed in url/title/tabs but screenshot is different → click opened a dropdown / popup / scrolled
- Nothing changed anywhere → click missed; try different coordinates, do NOT repeat the exact same click"""

    if warning:
        system_prompt += f"\n\n{warning}"

    active_model = model or VISION_MODEL

    if active_model.startswith("hf:"):
        # --- HuggingFace VL path ---
        model_id = active_model[3:]
        backend  = _get_hf_vl_backend(model_id)

        # Convert image bytes → PIL (prefer raw bytes; fall back to decoding base64)
        if raw_image_bytes:
            pil_image = Image.open(io.BytesIO(raw_image_bytes))
        else:
            pil_image = Image.open(io.BytesIO(base64.b64decode(base64_image)))

        raw_text = backend.generate(
            system_prompt=system_prompt,
            user_text="Look at the screenshot and current state. Output your JSON.",
            pil_image=pil_image,
            max_new_tokens=1024,
        )
    else:
        # --- Ollama path (default) ---
        response = ollama.chat(
            model=active_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': "Look at the screenshot and current state. Output your JSON.", 'images': [base64_image]}
            ],
            options={'temperature': 0.0, 'num_predict': 1024, 'num_ctx': 8192}
        )
        raw_text = response['message']['content'].strip()

    # Strip <think>...</think> blocks some instruct models prepend
    raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    _print(f"  [VISION-LLM] Raw: {raw_text}")

    def _normalize(d: dict) -> dict:
        """Some models (Holo2) nest the action fields inside an 'action' object:
        {"action": {"action": "type", "x":.., "y":.., "text":..}}
        Flatten that so decision['action'] is the string and x/y/text sit at top level —
        matching what the rest of the pipeline (and _get_field) expects.
        """
        if isinstance(d, dict) and isinstance(d.get("action"), dict):
            nested = d["action"]
            d["action"] = nested.get("action")
            for k, v in nested.items():
                if k != "action" and k not in d:
                    d[k] = v
        return d

    # Stage 1: normal full parse
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        try:
            return _normalize(json.loads(json_match.group(0)))
        except json.JSONDecodeError:
            pass

    # Stage 2: repair truncated JSON (model cut off mid-string)
    repair = re.sub(r',?\s*"[^"]*$', '', raw_text)
    repair = re.sub(r':\s*"[^"]*$', '', repair)
    repair = repair.rstrip(',').rstrip() + '}'
    try:
        parsed = json.loads(repair)
        _print(f"  [PARSER] Repaired truncated JSON.")
        return _normalize(parsed)
    except json.JSONDecodeError:
        pass

    # Stage 3: field-by-field salvage
    salvaged = {}
    for field in ('action', 'url', 'text', 'reasoning'):
        m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', raw_text)
        if m:
            salvaged[field] = m.group(1)
    for field in ('x', 'y'):
        m = re.search(rf'"{field}"\s*:\s*(\d+)', raw_text)
        if m:
            salvaged[field] = int(m.group(1))
    if 'action' in salvaged:
        _print(f"  [PARSER] Salvaged: {list(salvaged.keys())}")
        return _normalize(salvaged)

    _print("  [PARSER] All parse attempts failed.")
    return {"action": "error", "reasoning": "Failed to parse JSON."}

# --- 5. The Playwright Loop ---
def run_vision_agent(start_url, goal, max_steps=20, profile=None, manual_login=False, model=None):
    active_model = model or VISION_MODEL
    _print(f"\n{'='*50}")
    _print(f" PURE VISION AGENT DEPLOYED")
    _print(f" GOAL: '{goal}'")
    _print(f" MODEL: '{active_model}'")
    if profile:
        _print(f" PROFILE: '{profile}'")
    _print(f"{'='*50}\n")

    # Pre-load HF model BEFORE opening the browser so the download/load
    # doesn't happen mid-session while Playwright is already waiting.
    if active_model.startswith("hf:"):
        _get_hf_vl_backend(active_model[3:])

    # --- Goal refinement (one-time, before browser opens) ---
    strategy = refine_goal(goal, active_model)

    browser_args = ['--window-size=1920,1080', '--disable-infobars', '--force-device-scale-factor=1']

    with sync_playwright() as p:
        browser = None   # only set on non-persistent path; used to pick correct cleanup

        if profile:
            profile_dir = os.path.join(PROFILES_DIR, profile)
            os.makedirs(profile_dir, exist_ok=True)
            _print(f"  Loading profile from: {profile_dir}")
            # launch_persistent_context returns a BrowserContext directly, not a Browser
            context = p.chromium.launch_persistent_context(
                user_data_dir=profile_dir,
                headless=False,
                args=browser_args,
                viewport={'width': 1920, 'height': 1080},
                device_scale_factor=1
            )
            page = context.pages[0] if context.pages else context.new_page()
        else:
            browser = p.chromium.launch(headless=False, args=browser_args)
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                device_scale_factor=1
            )
            page = context.new_page()

        _print(f"  Navigating to initial state: {start_url} ...")
        page.goto(start_url, wait_until="domcontentloaded")

        if manual_login:
            _print("\n  [MANUAL LOGIN] Browser is open. Log in, set up cookies, do whatever you need.")
            _print("  [MANUAL LOGIN] When ready, press Enter here to start the agent...")
            input()
            _print("  [MANUAL LOGIN] Starting agent.\n")

        action_history = []
        scratchpad = {}          # structured dict — in-place state, not append log
        known_tab_count = 1
        step = 1

        try:
            while step <= max_steps:
                # --- Auto-follow new tabs ---
                # Amazon (and others) open product links in a new tab. The tab may open
                # slightly after our click handler's detection window. Check at the TOP
                # of every step — if new tabs appeared since last step, follow the newest.
                current_tab_count = len(context.pages)
                if current_tab_count > known_tab_count:
                    newest = context.pages[-1]
                    if newest != page:
                        page = newest
                        page.bring_to_front()
                        _print(f"  [TAB] New tab detected, auto-following → {page.url}")
                known_tab_count = current_tab_count

                _print(f"\n--- Step {step}/{max_steps} ---")
                
                # --- State Snapshot ---
                time.sleep(1.5)
                current_url, page_title = safe_page_state(page)
                _print(f"  [STATE] URL: {current_url}")
                _print(f"  [STATE] Title: {page_title}")

                # --- Tab inventory ---
                all_tabs = []
                for i, p in enumerate(context.pages):
                    try:
                        all_tabs.append({"index": i, "url": p.url, "title": p.title()})
                    except Exception:
                        all_tabs.append({"index": i, "url": "?", "title": "(loading)"})
                tabs_str = json.dumps(all_tabs)
                _print(f"  [TABS] {tabs_str}")

                # Model sees all tabs in state and uses switch_tab explicitly
                known_tab_count = len(context.pages)

                # --- Vision Capture ---
                _print("  [SYSTEM] Capturing viewport...")
                llm_b64, cv_bytes = capture_screenshot(page)

                # --- Loop Detection (framework-level, not model-level) ---
                loop_warning = _detect_loop(action_history, n=3)
                if loop_warning:
                    _print(f"  [LOOP] {loop_warning}")

                # --- Decide ---
                decision = get_vision_decision(llm_b64, goal, action_history, current_url, page_title, tabs_str, scratchpad, model=active_model, raw_image_bytes=cv_bytes, warning=loop_warning, strategy=strategy)
                
                action = decision.get("action")
                reasoning = decision.get("think", decision.get("reasoning", ""))
                _print(f"  [DECISION] Action: {action} | Think: {reasoning}")

                # --- Scratchpad update (merge partial update into persistent state) ---
                mem_update = decision.get("scratchpad")
                if mem_update and isinstance(mem_update, dict):
                    scratchpad.update(mem_update)
                    _print(f"  [MEMORY] {json.dumps(scratchpad)}")

                # --- Execute ---
                if action == "done":
                    conclusion = scratchpad.get("conclusion", reasoning)
                    _print(f"\n  ✅ MISSION ACCOMPLISHED!")
                    _print(f"  Answer: {conclusion}")
                    _print(f"  Final Memory: {json.dumps(scratchpad, indent=2)}")
                    _print(f"  Final URL: {current_url}")
                    break
                    
                elif action == "goto":
                    target_url = _get_field(decision, "url")
                    if target_url:
                        if not target_url.startswith("http"): target_url = "https://" + target_url
                        _print(f"  [EXEC] Navigating via Address Bar to: {target_url}")
                        safe_goto(page, target_url)
                        action_history.append({"step": step, "action": "goto", "url": target_url})
                
                elif action == "scroll":
                    direction = (_get_field(decision, "direction") or "down").lower()
                    if direction not in ("up", "down"):
                        direction = "down"
                    key = "PageDown" if direction == "down" else "PageUp"
                    _print(f"  [EXEC] Scrolling {direction} ...")
                    page.keyboard.press(key)
                    time.sleep(2.0)
                    action_history.append({"step": step, "action": "scroll", "direction": direction})

                elif action == "switch_tab":
                    tab_index = decision.get("tab_index", 0)
                    all_pages = context.pages
                    if 0 <= tab_index < len(all_pages):
                        page = all_pages[tab_index]
                        page.bring_to_front()
                        page.wait_for_load_state("domcontentloaded")
                        current_url = page.url
                        page_title = page.title()
                        _print(f"  [EXEC] Switched to tab {tab_index}: {current_url}")
                        action_history.append({"step": step, "action": "switch_tab", "tab_index": tab_index, "url": current_url})
                    else:
                        _print(f"  [FAIL] Tab index {tab_index} out of range (have {len(all_pages)} tabs)")
                        action_history.append({"step": step, "action": "switch_tab", "status": "FAILED"})
                    
                elif action == "click":
                    # _get_field handles both top-level x/y and nested action-specific-fields
                    rough_x = _get_field(decision, "x")
                    rough_y = _get_field(decision, "y")
                    if rough_x is None or rough_y is None:
                        rough_x, rough_y = _extract_coords(_get_field(decision, "target") or "")
                    if rough_x is None:
                        _print(f"  [FAIL] Could not parse coordinates from decision: {decision}")
                        action_history.append({"step": step, "action": "click", "status": "FAILED", "reason": "bad coords"})
                    else:
                        exact_x = int((int(rough_x) / 1000.0) * 1920)
                        exact_y = int((int(rough_y) / 1000.0) * 1080)
                        _print(f"  [EXEC] Click at ({exact_x}, {exact_y})")
                        count_before = len(context.pages)
                        page.mouse.click(exact_x, exact_y)
                        time.sleep(2.0)  # longer wait — new tabs can open late

                        # Detect new tabs by count — more reliable than id() comparison
                        new_pages = list(context.pages)[count_before:]
                        for p in new_pages:
                            try:
                                p.wait_for_load_state("domcontentloaded", timeout=5000)
                            except Exception:
                                pass

                        # Auto-follow if we caught a new tab in our window
                        if new_pages:
                            page = new_pages[0]
                            page.bring_to_front()
                            _print(f"  [TAB] Auto-followed new tab → {page.url}")

                        # Read post-action state safely
                        try:
                            url_after   = page.url
                            title_after = page.title()
                        except Exception:
                            url_after, title_after = "?", "(loading)"
                        tabs_after = len(context.pages)

                        # Factual change report — let the agent interpret with the next screenshot
                        def _trim(s, n=60):
                            s = str(s) if s else ""
                            return s if len(s) <= n else s[:n] + "..."

                        changes = {
                            "url":   f"{_trim(current_url)} → {_trim(url_after)}" if url_after != current_url else "no change",
                            "title": f"{_trim(page_title, 40)} → {_trim(title_after, 40)}" if title_after != page_title else "no change",
                            "tabs":  f"{count_before} → {tabs_after}" if tabs_after != count_before else "no change",
                        }

                        outcome = {
                            "step": step, "action": "click",
                            "coords": f"({exact_x},{exact_y})",
                            "changes": changes,
                        }
                        _print(f"  [OUTCOME] {outcome}")
                        action_history.append(outcome)
                        
                elif action == "type":
                    # _get_field handles both top-level "text" and nested "action-specific-fields"
                    text_to_type = _get_field(decision, "text") or ""
                    typed = False

                    if not text_to_type.strip():
                        _print(f"  [FAIL] type action has empty text — model output: {decision}")
                        action_history.append({"step": step, "action": "type", "status": "FAILED", "reason": "empty text"})
                    else:
                        url_before = page.url

                        # Strategy 1: coords from model → click there → fill + Enter
                        # Most direct. Works when model provides x,y (8B models always do).
                        rough_x = _get_field(decision, "x")
                        rough_y = _get_field(decision, "y")
                        if rough_x is None or rough_y is None:
                            rough_x, rough_y = _extract_coords(_get_field(decision, "target") or "")

                        if rough_x is not None:
                            exact_x = int((int(rough_x) / 1000.0) * 1920)
                            exact_y = int((int(rough_y) / 1000.0) * 1080)
                            _print(f"  [EXEC] Coord type at ({exact_x},{exact_y}): '{text_to_type}'")
                            page.mouse.click(exact_x, exact_y)
                            time.sleep(0.3)
                            page.keyboard.press("Control+a")
                            page.keyboard.type(text_to_type, delay=30)
                            page.keyboard.press("Enter")
                            typed = True

                        # Strategy 2: locator.fill() — pierces shadow DOM, no coords needed.
                        # Fallback for small models that don't output coordinates.
                        if not typed:
                            for loc in [
                                page.get_by_role("searchbox"),
                                page.get_by_role("combobox"),
                                page.locator('input[type="search"]'),
                                page.locator('input[type="text"]'),
                                page.locator('textarea'),
                            ]:
                                try:
                                    loc.first.fill(text_to_type, timeout=3000)
                                    page.keyboard.press("Enter")
                                    _print(f"  [EXEC] fill() type: '{text_to_type}'")
                                    typed = True
                                    break
                                except Exception:
                                    continue

                        if typed:
                            try:
                                page.wait_for_load_state("domcontentloaded", timeout=8000)
                            except Exception:
                                time.sleep(2.0)
                            action_history.append({
                                "step": step, "action": "type",
                                "text": text_to_type, "status": "OK",
                                "page_changed": page.url != url_before,
                                "url_after": page.url,
                            })
                            _print(f"  [OUTCOME] typed=OK page_changed={page.url != url_before}")
                        else:
                            _print(f"  [FAIL] Could not find any input to type into")
                            action_history.append({"step": step, "action": "type", "status": "FAILED", "reason": "no input found"})
                        
                elif action == "note":
                    # Legacy freetext note — writes into scratchpad["notes"] list, capped at 5
                    note_text = decision.get("text", "")
                    notes = scratchpad.get("notes", [])
                    notes.append(f"[step {step}] {note_text}")
                    scratchpad["notes"] = notes[-5:]   # keep only the 5 most recent
                    _print(f"  [NOTE] {note_text}")
                    action_history.append({"step": step, "action": "note", "text": note_text})

                else:
                    _print(f"  [WARN] Unknown or unhandled action: {action}")
                    action_history.append({"step": step, "action": str(action), "status": "FAILED"})
                    
                step += 1
                
        except KeyboardInterrupt:
            _print("\n  Manual stop.")
        finally:
            page.wait_for_timeout(3000)
            if browser:
                browser.close()   # non-persistent path
            else:
                context.close()   # persistent context path
            _print("  Browser closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Vision Browser Agent")
    parser.add_argument("--goal", required=True, help="Task for the agent to complete")
    parser.add_argument("--profile", default=None, help="Browser profile name (saved in .browser_profiles/)")
    parser.add_argument("--manual-login", action="store_true", help="Pause for manual login before agent starts")
    parser.add_argument("--start-url", default="about:blank", help="URL to open on launch")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum agent steps")
    parser.add_argument("--model", default=None, help="Ollama model to use (default: qwen3-vl:8b-instruct)")
    args = parser.parse_args()

    run_vision_agent(
        start_url=args.start_url,
        goal=args.goal,
        max_steps=args.max_steps,
        profile=args.profile,
        manual_login=args.manual_login,
        model=args.model,
    )
