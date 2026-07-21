"""
agent_vision_only.py -- Pure Vision Browser Agent

Architecture:
    1. Vision Perception: Takes a compressed 512x512 JPEG of the viewport.
    2. State Awareness: Passes URL and Page Title to the VLM to anchor its completion logic.
    3. LLM Decision: qwen3-vl predicts coordinates or navigation actions.
    4. OpenCV Sniper: Edge detection snaps rough VLM coordinates to exact button centers.
    5. Playwright Execution: Performs the click/type/scroll/goto actions.
"""

import cv2
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
VISION_MODEL = "qwen3-vl:8b-instruct" # Use your active vision model (8B or 4B)

def _print(msg):
    print(msg, flush=True)

# --- 1. Screenshot Capture ---
def capture_screenshot(page, resize_dim=(1024, 1024)):
    """Capture screenshot, return (base64_for_llm, raw_bytes_for_opencv)."""
    raw_bytes = page.screenshot(type='jpeg', quality=100)
    image = Image.open(io.BytesIO(raw_bytes))
    image.thumbnail(resize_dim, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", optimize=True, quality=85)
    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return b64_str, raw_bytes

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
def get_vision_decision(base64_image, goal, history, current_url, page_title):
    # Pass only recent history to prevent amnesia but keep prompt lean
    history_str = json.dumps(history[-6:], indent=2) if history else "[]"
    
    system_prompt = f"""You are an autonomous web agent that navigates purely by looking at screenshots.

GOAL: {goal}

CURRENT STATE:
- URL: {current_url}
- Page Title: {page_title}

PAST ACTIONS:
{history_str}

=== RULES ===
1. ADDRESS BAR (goto): If the URL is 'about:blank' or you need to start your search on a completely new website, you MUST use the "goto" action. You cannot click the physical browser address bar.
2. COMPLETION (done): BEFORE doing anything, check the CURRENT STATE. If the URL has changed to a consumption/detail page AND the goal constraints are met, output action "done".
3. TYPING: NEVER output 'click' for a search bar. ALWAYS output 'type'. The system will focus the box and type for you.
4. COORDINATES: Output coordinates for the center of the target element. Coordinates are 0-1000 normalized.
5. AMNESIA GUARD: Look at your PAST ACTIONS. Do NOT repeat the exact same action if it failed or if the page hasn't changed.

Output strict JSON ONLY:
For navigation: {{"action": "goto", "url": "https://www.amazon.in", "reasoning": "..."}}
For clicking: {{"action": "click", "target": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>", "reasoning": "..."}}
For typing: {{"action": "type", "target": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>", "text": "text to type", "reasoning": "..."}}
For scrolling: {{"action": "scroll", "reasoning": "..."}}
For completion: {{"action": "done", "reasoning": "..."}}

=== EXAMPLES OF CORRECT FORMAT ===
IMPORTANT: The scenarios below only show the JSON structure. X1, Y1, X2, and Y2 are placeholders for actual normalized numbers (0-1000).

Scenario: The goal is to find 'laptops'. You see an empty search bar at the top of the screen.
Output:
{{"reasoning": "I need to search for laptops. I will use the type action directly on the search bar so the system focuses it and enters the text.", "action": "type", "target": "<|box_start|>(X1,Y1),(X2,Y2)<|box_end|>", "text": "laptops"}}

Scenario: The goal is to open a menu. You see a 'Filters' button.
Output:
{{"reasoning": "I need to click the filters button to reveal the sorting options.", "action": "click", "target": "<|box_start|>(X1,Y1),(X2,Y2)<|box_end|>"}}
=== END EXAMPLES ===

Output strict JSON ONLY. MUST include the "reasoning" key FIRST.

"""
    
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': "Analyze the state and screenshot. What is the precise JSON action?", 'images': [base64_image]}
        ],
        options={'temperature': 0.0}
    )
    
    raw_text = response['message']['content'].strip()
    _print(f"  [VISION-LLM] Raw: {raw_text}")
    
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return {"action": "error", "reasoning": "Failed to parse JSON."}

# --- 5. The Playwright Loop ---
def run_vision_agent(start_url, goal, max_steps=20):
    _print(f"\n{'='*50}")
    _print(f" PURE VISION AGENT DEPLOYED")
    _print(f" GOAL: '{goal}'")
    _print(f"{'='*50}\n")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, 
            args=['--window-size=1920,1080', '--disable-infobars', '--force-device-scale-factor=1']
        )
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=1
        )
        page = context.new_page()
        
        _print(f"  Navigating to initial state: {start_url} ...")
        page.goto(start_url, wait_until="networkidle")
        
        action_history = []
        step = 1
        
        try:
            while step <= max_steps:
                _print(f"\n--- Step {step}/{max_steps} ---")
                
                # --- State Snapshot ---
                page.wait_for_load_state("domcontentloaded")
                time.sleep(1.5) 
                current_url = page.url
                page_title = page.title()
                _print(f"  [STATE] URL: {current_url}")
                _print(f"  [STATE] Title: {page_title}")
                
                # --- Vision Capture ---
                _print("  [SYSTEM] Capturing viewport...")
                llm_b64, cv_bytes = capture_screenshot(page)
                
                # --- Decide ---
                decision = get_vision_decision(llm_b64, goal, action_history, current_url, page_title)
                
                action = decision.get("action")
                reasoning = decision.get("reasoning", "")
                _print(f"  [DECISION] Action: {action} | Reason: {reasoning}")
                
                # --- Execute ---
                if action == "done":
                    _print(f"\n  ✅ MISSION ACCOMPLISHED! Reason: {reasoning}")
                    _print(f"  Final URL: {current_url}")
                    break
                    
                elif action == "goto":
                    target_url = decision.get("url")
                    if target_url:
                        if not target_url.startswith("http"): target_url = "https://" + target_url
                        _print(f"  [EXEC] Navigating via Address Bar to: {target_url}")
                        page.goto(target_url, wait_until="networkidle")
                        action_history.append({"step": step, "action": "goto", "url": target_url})
                
                elif action == "scroll":
                    _print("  [EXEC] Scrolling down ...")
                    page.keyboard.press("PageDown")
                    time.sleep(2.0)
                    action_history.append({"step": step, "action": "scroll"})
                    
                elif action == "click":
                    target_str = decision.get("target", "")
                    rough_x, rough_y = _extract_coords(target_str)
                    if rough_x is None:
                        _print(f"  [FAIL] Could not parse coordinates from: {target_str}")
                        action_history.append({"step": step, "action": "click", "status": "FAILED", "reason": "bad coords"})
                    else:
                        # 🛠️ THE FIX: Bypass OpenCV and map directly to 1080p canvas
                        exact_x = int((rough_x / 1000.0) * 1920)
                        exact_y = int((rough_y / 1000.0) * 1080)
                        
                        _print(f"  [EXEC] Trusting LLM Direct Click at ({exact_x}, {exact_y})")
                        page.mouse.click(exact_x, exact_y)
                        
                        # Add a tiny sleep to let dropdown animations finish rendering
                        time.sleep(0.5) 
                        action_history.append({"step": step, "action": "click", "coords": f"({exact_x},{exact_y})"})
                        
                elif action == "type":
                    target_str = decision.get("target", "")
                    text_to_type = decision.get("text", "")
                    rough_x, rough_y = _extract_coords(target_str)
                    if rough_x is None:
                        _print(f"  [FAIL] Could not parse coordinates for typing from: {target_str}")
                        action_history.append({"step": step, "action": "type", "status": "FAILED", "reason": "bad coords"})
                    else:
                        exact_x, exact_y = snap_to_element(cv_bytes, rough_x, rough_y)
                        _print(f"  [EXEC] Focus ({exact_x},{exact_y}) and type '{text_to_type}'")
                        page.mouse.click(exact_x, exact_y)
                        time.sleep(0.5)
                        page.keyboard.type(text_to_type, delay=50) # Delay bypasses SPA bot detection
                        time.sleep(0.5)
                        page.keyboard.press("Enter")
                        action_history.append({"step": step, "action": "type", "text": text_to_type})
                        
                else:
                    _print(f"  [WARN] Unknown or unhandled action: {action}")
                    action_history.append({"step": step, "action": str(action), "status": "FAILED"})
                    
                step += 1
                
        except KeyboardInterrupt:
            _print("\n  Manual stop.")
        finally:
            page.wait_for_timeout(3000)
            browser.close()
            _print("  Browser closed.")

if __name__ == "__main__":
    run_vision_agent(
        start_url="about:blank",
        goal="Navigate to Amazon.in and search for 'protein powder', sort the results from lowest to highest price, and find the cheapest actual protein supplement available with 4 stars or higher, and give me its link. Before confirming just cross verify it.",
        max_steps=20
    )