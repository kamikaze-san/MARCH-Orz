import cv2
import numpy as np
import ollama
import re
import json
import time
from ocr_sniper import find_text_and_click
from browser_engine import VisionBrowser

# --- 1. The OpenCV Sniper (Unchanged) ---
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
    
    if not contours: return rough_x, rough_y
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        local_cx, local_cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(largest_contour)
        local_cx, local_cy = x + w // 2, y + h // 2
        
    return x_min + local_cx, y_min + local_cy

# --- 2. The 4B Agent Brain (Upgraded for JSON) ---
def get_next_action(goal, base64_image, history):
    history_str = json.dumps(history, indent=2) if history else "[]"
    
    # 1. Load the vault keys (not the values!)
    try:
        with open("vault.json", "r") as f:
            vault_keys = list(json.load(f).keys())
    except FileNotFoundError:
        vault_keys = []
    
    system_prompt = f"""You are an autonomous web agent.
Goal: {goal}

Available Secure Data Keys: {vault_keys}

Past Actions History:
{history_str}

Look at the screen, read your history, and determine the SINGLE next action.
CRITICAL RULES: 
1. STATE COMPARISON: Look at your 'Past Actions History'. If your last action was 'type', look at the NEW image. If a dropdown, suggestion list, or popup appeared directly below or near where you typed, your NEXT action MUST be to click that suggestion.
2. COMPLETION CRITERIA: A 'type' action is not finished until you have selected a value from the resulting dropdown (if one appears).
3. SEARCH FLOW: After selecting a location or clearing a popup, your next goal is to find the search bar, type the query, and press Enter.
4. FINAL REPORTING: Once you see the final product grid, use 'done' and describe every item's name and its stock status (e.g., "Amul Whey: In Stock, Amul Buttermilk: SOLD OUT").
You MUST output strictly in JSON format. You MUST include a "reasoning" key FIRST.
Choose ONE format ONLY:
{{"reasoning": "...", "action": "ocr_click", "target": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>", "text": "Exact text"}}
{{"reasoning": "...", "action": "click", "target": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>"}}
{{"reasoning": "...", "action": "type", "target": "<|box_start|>(x1,y1),(x2,y2)<|box_end|>", "text": "exact text or $key"}}
{{"reasoning": "...", "action": "done", "summary": "Write your actual detailed findings here based on the screen."}}
"""
    
    response = ollama.chat(
        model='qwen3-vl:8b-instruct',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': "What is the next JSON action?", 'images': [base64_image]}
        ],
        options={'temperature': 0.0}
    )
    
    raw_text = response['message']['content'].strip()
    print(f"\n   [AI Raw Output]: {raw_text}")
    
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return {"action": "error", "message": "Failed to parse JSON."}

# --- 3. The Autonomous Loop ---
def run_autonomous_agent(start_url, goal, max_steps=10):
    agent = VisionBrowser(headless=False)
    agent.navigate(start_url)
    
    print(f"\n========================================")
    print(f" AGENT DEPLOYED: '{goal}'")
    print(f"========================================\n")
    
    action_history = [] # THE NEW MEMORY SCRATCHPAD
    step = 1
    
    while step <= max_steps:
        print(f"--- Step {step} ---")
        time.sleep(1.5) 
        
        print("Capturing screen state...")
        llm_b64, cv_bytes = agent.capture_vision_state()
        
        # 2. Decide
        print("Thinking...")
        action_obj = get_next_action(goal, llm_b64, action_history) 
        action_type = action_obj.get("action")
        
        # 3. Act
        if action_type == "click":
            target_str = action_obj.get("target", "")
            # 1. The Official Prompt Format
            match = re.search(r'<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>', target_str)
            
            # 2. The Raw Array Format
            if not match: match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', target_str)
            
            # 3. The C++ / Windows format hallucination
            if not match: match = re.search(r'CRect\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
            
            # 4. The Apple iOS format hallucination
            if not match: match = re.search(r'CGRect\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)

            # 5. The Tool-Calling Hallucination
            if not match: match = re.search(r'<tool_response>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)[\s\n]*\(\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                rough_x, rough_y = (x1 + x2) // 2, (y1 + y2) // 2
                exact_x, exact_y = snap_to_element(cv_bytes, rough_x, rough_y)
                agent.click(exact_x, exact_y)
                
                # INJECT EXACT MEMORY
                action_history.append({"action": "click", "target": target_str, "status": "completed"})
            else:
                print("Error: Could not extract coordinates.")
                
        elif action_type == "type":
            
            target_str = action_obj.get("target", "")
            raw_text = action_obj.get("text", "")
            
            # 1. Resolve Vault Data BEFORE touching the browser
            if raw_text.startswith("$"):
                key = raw_text[1:]
                try:
                    with open("vault.json", "r") as f:
                        data_vault = json.load(f)
                    if key in data_vault:
                        text_to_type = data_vault[key]
                        print(f"🔒 Vault Unlocked: Injecting secure data for '{key}'")
                        memory_log = f"Secure Data Inserted: {key}"
                    else:
                        print(f"⚠️ Error: Missing vault key {key}. Typing literal string.")
                        text_to_type = raw_text
                        memory_log = raw_text
                except Exception as e:
                    text_to_type = raw_text
                    memory_log = raw_text
            else:
                text_to_type = raw_text
                memory_log = raw_text

            # 2. Extract Coordinates (Indestructible Regex)
            match = re.search(r'<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>', target_str)
            if not match: match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', target_str)
            if not match: match = re.search(r'CRect\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
            if not match: match = re.search(r'CGRect\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
            if not match: match = re.search(r'<tool_response>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)[\s\n]*\(\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
            
            # 3. Execute EXACTLY ONCE
            # 3. Execute EXACTLY ONCE with correct math
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                qwen_x, qwen_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # THE MISSING MATH: Convert Qwen's 0-1000 scale to real screen pixels
                nparr = np.frombuffer(cv_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                height, width = img.shape[:2]
                
                real_x = int((qwen_x / 1000.0) * width)
                real_y = int((qwen_y / 1000.0) * height)
                
                print(f"Executing focus click at real pixel X:{real_x}, Y:{real_y}")
                agent.click(real_x, real_y) # Focus the actual box
                agent.type_text(text_to_type)
                
                # Give the site a moment to trigger the JS dropdown
                print("Waiting for UI to stabilize...")
                agent.page.wait_for_timeout(2500) 
                
                
                
                action_history.append({"action": "type", "text": memory_log, "status": "completed"})
            else:
                print("Error: Could not extract bounding box for type action.")
        elif action_type == "ocr_click":
            target_str = action_obj.get("target", "")
            text_to_find = action_obj.get("text", "")
            
            # The indestructible regex to extract the rough coordinate
            match = re.search(r'<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>', target_str)
            if not match: match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', target_str)
            if not match: match = re.search(r'CRect\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
            if not match: match = re.search(r'CGRect\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
            if not match: match = re.search(r'<tool_response>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)[\s\n]*\(\s*(\d+)\s*,\s*(\d+)\s*\)', target_str)
            
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                rough_x, rough_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                print(f"Deploying Bounded OCR Sniper for: '{text_to_find}' near X:{rough_x}, Y:{rough_y}...")
                
                # Pass the LLM's rough coordinate to Tesseract
                exact_x, exact_y = find_text_and_click(cv_bytes, text_to_find, rough_x, rough_y)
                
                if exact_x and exact_y:
                    agent.click(exact_x, exact_y)
                    action_history.append({"action": "ocr_click", "text": text_to_find, "status": "completed"})
                else:
                    print(f"⚠️ OCR Blindness! Falling back to the AI's visual coordinate at X:{rough_x}, Y:{rough_y}")
                    #agent.click(rough_x, rough_y)
                    #action_history.append({"action": "ocr_click", "text": text_to_find, "status": "completed via AI fallback target"})
            else:
                print("Error: Could not extract bounding box for ocr_click.")

        elif action_type == "done":
            summary = action_obj.get("summary", "No summary provided.")
            print("\n✅ MISSION ACCOMPLISHED! Here is the Intel:")
            print(f"========================================\n{summary}\n========================================")
            break
            
        else:
            print(f"\n❌ AI hallucinated an unknown action: {action_type}")
            print(f"Full object: {action_obj}")
            break # Kills the loop so you don't waste 10 steps doing nothing
            
        step += 1

    agent.page.wait_for_timeout(5000)
    agent.close()

if __name__ == "__main__":
    # The ultimate test: A multi-step sequence
    # The true autonomous test
    test_goal = "Type my pincode and then Search the store for protein. Scan the product grid and use the 'done' action to give me a summary of exactly which protein items are currently In Stock and which are Sold Out."
    
    run_autonomous_agent("https://shop.amul.com", test_goal, max_steps=15)