"""
agent_hybrid.py -- Hybrid DOM + Vision Browser Agent

DOM-first perception, vision fallback when DOM can't find the target.
Self-contained: no imports from memory.py, planner.py, policy.py.
Simple action_history list for state.

Flow per step:
    1. Extract DOM elements (Playwright JS)
    2. Read URL + page title (always available)
    3. Check completion via URL (robust, like dom_test)
    4. Ask LLM: pick element by ID, scroll, or request vision_fallback
    5. If vision_fallback: screenshot → vision LLM full decision (click/type/scroll/done)
    6. Append to action_history
"""

import cv2
import numpy as np
import ollama
import requests
import json
import re
import time
import base64
import io
import os
from PIL import Image
from playwright.sync_api import sync_playwright

os.environ["PYTHONUNBUFFERED"] = "1"

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DOM_MODEL = "qwen3-vl:8b-instruct"      # Text-only DOM decisions
VISION_MODEL = "qwen3-vl:8b-instruct"   # Vision fallback (coordinate output)


def _print(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------
# 1. DOM Extraction (from dom_test.py)
# ---------------------------------------------------------------

def extract_viewport_elements(page):
    """Extract visible interactive elements and tag them with data-ai-id."""
    js_code = """
    () => {
        document.querySelectorAll('[data-ai-id]').forEach(el => el.removeAttribute('data-ai-id'));
        let elements = [];
        let idCounter = 1;
        let allNodes = document.querySelectorAll(
    'a, button, input:not([type="hidden"]), textarea, select, [role="button"], [role="tab"], [role="menuitem"]'
);

        allNodes.forEach(node => {
            let rect = node.getBoundingClientRect();
            let isVisible = (
                rect.top >= 0 && rect.left >= 0 &&
                rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                rect.right <= (window.innerWidth || document.documentElement.clientWidth) &&
                rect.width > 0 && rect.height > 0
            );

            let style = window.getComputedStyle(node);
            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') isVisible = false;

            if (isVisible) {
                let tag = node.tagName.toLowerCase();
                let text = '';

                // For input/textarea: grab placeholder or value since innerText is empty
                if (tag === 'input' || tag === 'textarea') {
                    text = node.value || node.placeholder || node.getAttribute('aria-label') || '';
                    let inputType = node.type || 'text';
                    tag = tag + '[' + inputType + ']';
                } else {
                    text = (node.innerText || node.getAttribute('aria-label') || node.title || '').trim();
                }

                text = text.replace(/\\n/g, ' ').trim();
                if (text.length > 0) {
                    node.setAttribute('data-ai-id', idCounter);
                    elements.push(`[${idCounter}] ${tag}: '${text.substring(0, 75)}'`);
                    idCounter++;
                }
            }
        });
        return elements.join('\\n');
    }
    """
    return page.evaluate(js_code)


# ---------------------------------------------------------------
# 2. Screenshot Capture (for vision fallback)
# ---------------------------------------------------------------

def capture_screenshot(page, resize_dim=(512, 512)):
    """Capture screenshot, return (base64_for_llm, raw_bytes_for_opencv)."""
    raw_bytes = page.screenshot(type='jpeg', quality=100)
    image = Image.open(io.BytesIO(raw_bytes))
    image.thumbnail(resize_dim, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", optimize=True, quality=85)
    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return b64_str, raw_bytes


# ---------------------------------------------------------------
# 3. OpenCV Sniper (from V1, for vision fallback)
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
# 4. Coordinate Extraction (handles all hallucination formats)
# ---------------------------------------------------------------

def _extract_coords(target_str):
    if not target_str:
        return None, None

    match = re.search(r'<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>', target_str)
    if not match: match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', target_str)
    if not match: match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)

    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1 + x2) // 2, (y1 + y2) // 2

    single = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
    if single:
        x, y = map(int, single.groups())
        return x, y

    return None, None


# ---------------------------------------------------------------
# 5. DOM-First LLM Decision (structured schema, no image)
# ---------------------------------------------------------------

def get_dom_decision(dom_elements, goal, history, current_url, page_title):
    """Text-only structured LLM call. Returns action dict."""
    history_str = json.dumps(history[-8:], indent=2) if history else "[]"

    prompt = f"""You are an autonomous web navigation agent.

GOAL: {goal}

CURRENT STATE:
- URL: {current_url}
- Page Title: {page_title}

PAST ACTIONS (recent):
{history_str}

=== STEP 1: CHECK IF GOAL IS ALREADY COMPLETE ===
BEFORE looking at any elements, answer this:
- Has the URL CHANGED to a consumption page (e.g. /watch for videos, /product for items, /article for reading)?
- Does the PAGE TITLE show the specific target item's name (not a list or search page)?

CRITICAL DISTINCTION:
- SEEING the target in an element list is NOT completion. That means you must CLICK it.
- CONSUMING the target (URL changed to a player/viewer/detail page) IS completion.
- If the URL still shows a list page (/videos, /results, /search, /channel), the goal is NOT complete.

Only set "is_goal_complete" to true when the URL confirms you are ON the target's own page.

=== STEP 2: ONLY IF GOAL IS NOT COMPLETE, NAVIGATE ===
VISIBLE INTERACTIVE ELEMENTS:
{dom_elements}

Navigation rules:
- If the target item IS in the element list: click it by element_id.
- If the target is NOT visible but you are on the right page: scroll to reveal more.
- If you are on the wrong page: click a navigation link.
- If you need to enter text: use "type" with element_id of the input field.
- If the target is NOT in the element list and scrolling won't help (e.g. it's an image/thumbnail with no text label): use "vision_fallback" and describe what to look for.
- Do NOT repeat a failed action. Try something different.
- Keep reasoning under 2 sentences.
- THE ADDRESS BAR (goto): If you need to start a search on a completely new website (e.g., Amazon, Google) and you are not currently on that site, you MUST use the "goto" action and provide the full "url". 
- NEVER use "type" to enter a website URL. The "type" action is STRICTLY for typing text into search boxes or forms that have a visible element_id.
"""

    schema = {
        "type": "object",
        "properties": {
            "completion_check": {
                "type": "string",
                "description": "FIRST: State the current URL type — consumption page (/watch, /product) or list page (/videos, /results, /channel). A list page = NOT complete."
            },
            "is_goal_complete": {
                "type": "boolean",
                "description": "true ONLY if the URL is a consumption/detail page. false if URL is still a list/search/channel page."
            },
            "reasoning": {
                "type": "string",
                "description": "1-2 sentence explanation of next action."
            },
            "action": {
                "type": "string",
                "enum": ["click", "type", "scroll", "done", "vision_fallback", "goto"],
                "description": "Use 'goto' for address bar navigation. If is_goal_complete is true, MUST be 'done'."
            },
            "url": {
                "type": ["string", "null"],
                "description": "The full URL to navigate to if action is 'goto' (e.g., 'https://www.amazon.in'). null otherwise."
            },
            "element_id": {
                "type": ["integer", "null"],
                "description": "ID of element for click/type. null for scroll/done/vision_fallback."
            },
            "text": {
                "type": ["string", "null"],
                "description": "Text for type actions. null otherwise."
            },
            "vision_description": {
                "type": ["string", "null"],
                "description": "For vision_fallback only: describe the visual element to find on screen. null otherwise."
            }
        },
        "required": ["completion_check", "is_goal_complete", "reasoning", "action"]
    }

    payload = {
        "model": DOM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0.0}
    }

    _print("  [DOM] Thinking ...")
    response = requests.post(OLLAMA_ENDPOINT, json=payload)

    try:
        raw_data = response.json()
        text_output = raw_data.get("response", "")
        result = json.loads(text_output)
        _print(f"  [DOM] Completion: {result.get('completion_check', '')}")
        _print(f"  [DOM] Goal done: {result.get('is_goal_complete', False)}")
        _print(f"  [DOM] Reasoning: {result.get('reasoning', '')}")
        _print(f"  [DOM] Action: {result.get('action')} | ID: {result.get('element_id')} | Vision: {result.get('vision_description', '')}")
        return result
    except Exception as e:
        _print(f"  [DOM] Parse error: {e}")
        return None


# ---------------------------------------------------------------
# 6. Vision Fallback — Full Decision-Maker (same actions as DOM)
# ---------------------------------------------------------------

def get_vision_decision(base64_image, goal, history, current_url, page_title, hint=""):
    """
    Vision LLM call. Same actions as DOM path (click, type, scroll, done)
    but uses screenshot instead of DOM elements. Returns coordinates for
    click/type targets. Always receives URL + title for state awareness.
    """
    history_str = json.dumps(history[-6:], indent=2) if history else "[]"
    hint_line = f"\nHINT from DOM agent: {hint}" if hint else ""

    system_prompt = f"""You are an autonomous web agent that navigates by looking at screenshots.

GOAL: {goal}

CURRENT STATE:
- URL: {current_url}
- Page Title: {page_title}

PAST ACTIONS:
{history_str}{hint_line}

=== COMPLETION CHECK ===
If the URL is a consumption page (/watch, /product, /article) and the page title matches the goal target, output action "done".
If the URL is still a list/search/channel page, the goal is NOT complete.

=== NAVIGATION ===
Look at the screenshot and decide the SINGLE next action.
Output strict JSON:

For clicking: {{"action": "click", "target": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>", "reasoning": "..."}}
For typing: {{"action": "type", "target": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>", "text": "text to type", "reasoning": "..."}}
For scrolling: {{"action": "scroll", "reasoning": "..."}}
For done: {{"action": "done", "reasoning": "..."}}

RULES:
- NEVER output 'click' for a search bar or text input field. ALWAYS output 'type' with the 'text' you want to enter. The system will click it for you automatically.
- Coordinates are 0-1000 normalized. Be precise about what you see.
- Do NOT repeat failed actions. If you clicked something and nothing happened, try something else.
"""

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'What is the next action?', 'images': [base64_image]}
        ],
        options={'temperature': 0.0}
    )

    raw_text = response['message']['content'].strip()
    _print(f"  [VISION] Raw: {raw_text}")

    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return {"action": "error", "reasoning": "Vision LLM failed to produce valid JSON."}


# ---------------------------------------------------------------
# 7. The Hybrid Loop
# ---------------------------------------------------------------

def run_hybrid_agent(start_url, goal, max_steps=15):
    _print(f"\n========================================")
    _print(f" HYBRID AGENT: '{goal}'")
    _print(f"========================================\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                '--window-size=1920,1080',
                '--disable-infobars',
                '--force-device-scale-factor=1'
            ]
        )
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=1
        )
        page = context.new_page()
        _print(f"  Browser 1920x1080. Navigating to {start_url} ...")
        page.goto(start_url, wait_until="networkidle")

        action_history = []
        step = 1

        try:
            while step <= max_steps:
                _print(f"\n--- Step {step}/{max_steps} ---")

                # -- Always available: URL + title --
                page.wait_for_load_state("domcontentloaded")
                time.sleep(1.5)
                current_url = page.url
                page_title = page.title()
                _print(f"  [STATE] URL: {current_url}")
                _print(f"  [STATE] Title: {page_title}")

                # -- DOM extraction --
                dom_elements = extract_viewport_elements(page)

                # -- DOM-first decision --
                decision = get_dom_decision(dom_elements, goal, action_history, current_url, page_title)

                if not decision:
                    _print("  [WARN] Decision failed, retrying ...")
                    action_history.append({"step": step, "action": "error", "status": "FAILED"})
                    step += 1
                    continue

                action = decision.get("action")
                element_id = decision.get("element_id")
                goal_done = decision.get("is_goal_complete", False)

                # ============================================
                # DONE — goal achieved (URL-based check passed)
                # ============================================
                if action == "done" or goal_done:
                    _print("\n  ✅ MISSION ACCOMPLISHED!")
                    _print(f"  URL: {current_url}")
                    _print(f"  Title: {page_title}")
                    time.sleep(5)
                    break
                # ============================================
                # GOTO — Virtual Address Bar
                # ============================================
                elif action == "goto":
                    target_url = decision.get("url")
                    if target_url:
                        # Ensure it has http/https so Playwright doesn't crash
                        if not target_url.startswith("http"):
                            target_url = "https://" + target_url
                        
                        _print(f"  [EXEC] Navigating via address bar to: {target_url}")
                        try:
                            page.goto(target_url, wait_until="networkidle")
                            action_history.append({
                                "step": step, "action": "goto",
                                "url": target_url, "status": "success"
                            })
                        except Exception as e:
                            _print(f"  [EXEC] Goto failed: {e}")
                            action_history.append({
                                "step": step, "action": "goto",
                                "url": target_url, "status": "FAILED", "error": str(e)
                            })
                    else:
                        _print("  [EXEC] Goto failed: No URL provided by LLM.")    
                # ============================================
                # CLICK — DOM native via element ID
                # ============================================
                elif action == "click" and element_id:
                    try:
                        locator = page.locator(f"[data-ai-id='{element_id}']")
                        locator.click(timeout=3000)
                        _print(f"  [EXEC] Clicked element #{element_id}")
                        action_history.append({
                            "step": step, "action": "click",
                            "id": element_id, "status": "success",
                            "url_after": page.url
                        })
                    except Exception as e:
                        _print(f"  [EXEC] Click failed: {e}")
                        action_history.append({
                            "step": step, "action": "click",
                            "id": element_id, "status": "FAILED", "error": str(e)
                        })

                # ============================================
                # TYPE — DOM native via element ID + Enter
                # ============================================
                # ============================================
                # TYPE — DOM native via element ID + Enter
                # ============================================
                elif action == "type" and element_id:
                    text_to_type = decision.get("text", "")
                    try:
                        locator = page.locator(f"[data-ai-id='{element_id}']")
                        
                        # 🛠️ THE FIX: Avoid .fill(). Click the box, wait for the JS framework to 
                        # register the focus, then type human-like keystrokes.
                        locator.click(timeout=3000)
                        time.sleep(0.5) 
                        page.keyboard.type(text_to_type, delay=50)
                        time.sleep(0.5)
                        page.keyboard.press("Enter")
                        
                        _print(f"  [EXEC] Typed '{text_to_type}' into #{element_id} + Enter")
                        action_history.append({
                            "step": step, "action": "type",
                            "text": text_to_type, "status": "success"
                        })
                    except Exception as e:
                        _print(f"  [EXEC] Type failed: {e}")
                        action_history.append({
                            "step": step, "action": "type",
                            "text": text_to_type, "status": "FAILED", "error": str(e)
                        })

                # ============================================
                # SCROLL — PageDown
                # ============================================
                elif action == "scroll":
                    _print("  [EXEC] Scrolling down ...")
                    page.keyboard.press("PageDown")
                    time.sleep(2.0)
                    action_history.append({
                        "step": step, "action": "scroll", "url": current_url
                    })

                # ============================================
                # VISION FALLBACK — full decision via screenshot
                # ============================================
                elif action == "vision_fallback":
                    hint = decision.get("vision_description", "")
                    _print(f"  [VISION] Switching to vision mode{' (hint: ' + hint + ')' if hint else ''} ...")
                    llm_b64, cv_bytes = capture_screenshot(page)

                    v_decision = get_vision_decision(
                        llm_b64, goal, action_history,
                        current_url, page_title, hint=hint
                    )

                    v_action = v_decision.get("action")
                    v_reasoning = v_decision.get("reasoning", "")
                    _print(f"  [VISION] Action: {v_action} | Reason: {v_reasoning}")

                    # -- Vision: done --
                    if v_action == "done":
                        _print("\n  ✅ MISSION ACCOMPLISHED! (via vision)")
                        _print(f"  URL: {current_url}")
                        _print(f"  Title: {page_title}")
                        time.sleep(5)
                        break

                    # -- Vision: scroll --
                    elif v_action == "scroll":
                        _print("  [VISION] Scrolling down ...")
                        page.keyboard.press("PageDown")
                        time.sleep(2.0)
                        action_history.append({
                            "step": step, "action": "vision_scroll",
                            "url": current_url, "status": "success"
                        })

                    # -- Vision: click --
                    elif v_action == "click":
                        target_str = v_decision.get("target", "")
                        rough_x, rough_y = _extract_coords(target_str)
                        if rough_x is None:
                            _print(f"  [VISION] Bad coords: {target_str}")
                            action_history.append({
                                "step": step, "action": "vision_click",
                                "status": "FAILED", "reason": "bad coords"
                            })
                        else:
                            exact_x, exact_y = snap_to_element(cv_bytes, rough_x, rough_y)
                            _print(f"  [VISION] Clicking at ({exact_x}, {exact_y})")
                            page.mouse.click(exact_x, exact_y)
                            action_history.append({
                                "step": step, "action": "vision_click",
                                "coords": f"({exact_x},{exact_y})",
                                "status": "success"
                            })

                    # -- Vision: type --
                    elif v_action == "type":
                        target_str = v_decision.get("target", "")
                        text_to_type = v_decision.get("text", "")
                        rough_x, rough_y = _extract_coords(target_str)
                        if rough_x is None:
                            _print(f"  [VISION] Bad coords for type: {target_str}")
                            action_history.append({
                                "step": step, "action": "vision_type",
                                "status": "FAILED", "reason": "bad coords"
                            })
                        else:
                            exact_x, exact_y = snap_to_element(cv_bytes, rough_x, rough_y)
                            _print(f"  [VISION] Focus ({exact_x},{exact_y}), type '{text_to_type}'")
                            page.mouse.click(exact_x, exact_y)
                            page.keyboard.type(text_to_type, delay=50)
                            page.keyboard.press("Enter")
                            action_history.append({
                                "step": step, "action": "vision_type",
                                "text": text_to_type, "status": "success"
                            })

                    else:
                        _print(f"  [VISION] Unknown action: {v_action}")
                        action_history.append({
                            "step": step, "action": f"vision_{v_action}",
                            "status": "FAILED"
                        })

                else:
                    _print(f"  [WARN] Unhandled action: {action}")
                    action_history.append({
                        "step": step, "action": str(action), "status": "FAILED"
                    })

                step += 1

        except KeyboardInterrupt:
            _print("\n  Manual stop.")
        finally:
            page.wait_for_timeout(3000)
            browser.close()
            _print("  Browser closed.")


# ---------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------

if __name__ == "__main__":
    run_hybrid_agent(
        start_url="about:blank",
        goal="Navigate to Amazon.in and search for 'protein powder', after that, using the vision mode find that button from which you can sort the results from lowest to highest price, and find the cheapest actual protein supplement available.",
        max_steps=15
    )
