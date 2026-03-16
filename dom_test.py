import json
import requests
import time
import re
from playwright.sync_api import sync_playwright

# --- CONFIGURATION ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-vl:8b-instruct" # Your local model



def get_llm_decision(compressed_dom, goal, history, current_url, page_title):
    history_str = json.dumps(history[-6:], indent=2) if history else "[]"
    
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
- If the URL still shows a list page (/videos, /results, /search, /channel), the goal is NOT complete even if you can see the target.

Only set "is_goal_complete" to true when the URL confirms you are ON the target's own page.
DO NOT look at sidebar recommendations or suggested content. Those are distractions.

=== STEP 2: ONLY IF GOAL IS NOT COMPLETE, NAVIGATE ===
VISIBLE INTERACTIVE ELEMENTS:
{compressed_dom}

Navigation rules:
- If the target item is visible in the element list: click it.
- If the target is NOT visible but you are on the right page: scroll to reveal more content.
- If you are on the wrong page entirely: click a navigation link to get closer.
- If you need to enter text: use "type".
- Do NOT repeat a failed action. Try something different.
- Keep reasoning under 2 sentences.
"""

    schema = {
        "type": "object",
        "properties": {
            "completion_check": {
                "type": "string",
                "description": "FIRST: State the current URL. Is it a consumption page (/watch, /product, /article) or a list page (/videos, /results, /channel)? A list page means NOT complete."
            },
            "is_goal_complete": {
                "type": "boolean",
                "description": "true ONLY if the URL is a consumption/detail page (like /watch). false if URL is still a list/search/channel page, even if the target is visible in the list."
            },
            "reasoning": {
                "type": "string",
                "description": "If goal is not complete: 1-2 sentence explanation of what to do next."
            },
            "action": {
                "type": "string",
                "enum": ["click", "type", "scroll", "done"],
                "description": "If is_goal_complete is true, this MUST be 'done'. Otherwise pick click/type/scroll."
            },
            "element_id": {
                "type": ["integer", "null"],
                "description": "ID of element to interact with. null for scroll/done."
            },
            "text": {
                "type": ["string", "null"],
                "description": "Text for type actions. null otherwise."
            }
        },
        "required": ["completion_check", "is_goal_complete", "reasoning", "action"]
    }
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {
            "temperature": 0.0
        }
    }
    
    print("\n🧠 Thinking (Structured Output Mode)...")
    response = requests.post(OLLAMA_ENDPOINT, json=payload)
    
    try:
        raw_data = response.json()
        text_output = raw_data.get("response", "")
        
        # Because we used a schema, the output is guaranteed to be pure JSON.
        # No regex or string splitting needed.
        return json.loads(text_output)
        
    except Exception as e:
        print(f"❌ Parse error: {e}")
        print(f"Raw response: {response.text}")
        return None
        
def extract_viewport_elements(page):
    js_code = """
    () => {
        // 🧹 THE FIX: Wipe all old AI IDs from the previous step!
        document.querySelectorAll('[data-ai-id]').forEach(el => el.removeAttribute('data-ai-id'));

        let elements = [];
        let idCounter = 1;
        let allNodes = document.querySelectorAll('a, button, input, [role="button"], [role="tab"]');
        
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
                let text = (node.innerText || node.getAttribute('aria-label') || node.title || '').trim().replace(/\\n/g, ' ');
                if (text.length > 0) {
                    node.setAttribute('data-ai-id', idCounter);
                    elements.push(`[${idCounter}] ${node.tagName.toLowerCase()}: '${text.substring(0, 75)}'`);
                    idCounter++;
                }
            }
        });
        return elements.join('\\n');
    }
    """
    return page.evaluate(js_code)

def run_dom_agent():
    goal = "Go to pewdiepie's channel, click on the videos tab, find a video about linux and click on it"
    action_history = []
    step = 1
    max_steps = 10
    
    with sync_playwright() as p:
        # --- FIX: Force the physical window to be 1080p, and disable info bars ---
        print("🌐 Browser initialized at strict 1920x1080.")
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
        page.goto("https://www.youtube.com/results?search_query=PewDiePie")
        try:
            while step <= max_steps:
                print(f"\n--- STEP {step} ---")
                
                # 1. Wait for UI to stabilize
                # 1. Wait for UI to stabilize
                page.wait_for_load_state("domcontentloaded")
                time.sleep(2.5) 
                
                # 2. Extract DOM & Macro State
                current_url = page.url
                page_title = page.title()
                compressed_dom = extract_viewport_elements(page)
                
                print(f"📍 Current URL: {current_url}")
                print(f"📍 Page Title: {page_title}")
                
                # 3. Get LLM Decision (Pass the new variables)
                decision = get_llm_decision(compressed_dom, goal, action_history, current_url, page_title)
                
                if not decision:
                    print("⚠️ Failed to get decision, retrying...")
                    step += 1
                    continue
                    
                action = decision.get("action")
                element_id = decision.get("element_id")
                reasoning = decision.get("reasoning", "")
                completion = decision.get("completion_check", "")
                goal_done = decision.get("is_goal_complete", False)
                
                print(f"🔍 Completion Check: {completion}")
                print(f"🏁 Goal Complete: {goal_done}")
                print(f"🤖 Reasoning: {reasoning}")
                print(f"⚡ Action: {action} on ID: {element_id}")
                
                # 4. Execute Native Playwright Actions
                if action == "click" and element_id:
                    locator = page.locator(f"[data-ai-id='{element_id}']")
                    locator.click(timeout=3000)
                    action_history.append({"step": step, "action": "click", "id": element_id, "status": "success"})

                elif action == "scroll":
                    print("⏬ Scrolling down the page...")
                    # PageDown perfectly mimics a human scrolling and triggers lazy-loading
                    page.keyboard.press("PageDown")
                    # We need a slightly longer sleep here because YouTube has to fetch new thumbnails
                    time.sleep(2.5) 
                    action_history.append({"step": step, "action": "scroll", "url": current_url})

                elif action == "type" and element_id:
                    text_to_type = decision.get("text", "")
                    locator = page.locator(f"[data-ai-id='{element_id}']")
                    locator.fill(text_to_type)
                    locator.press("Enter")
                    action_history.append({"step": step, "action": "type", "text": text_to_type, "status": "success"})
                    
                elif action == "done":
                    print("\n✅ MISSION ACCOMPLISHED!")
                    print(f"Summary: {decision.get('summary', 'Task complete.')}")
                    time.sleep(10) # Let the video play for a bit!
                    break
                    
                step += 1
                
        except KeyboardInterrupt:
            print("\n🛑 Manual Stop Triggered.")
        finally:
            browser.close()

if __name__ == "__main__":
    run_dom_agent()