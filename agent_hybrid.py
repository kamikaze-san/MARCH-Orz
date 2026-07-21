"""
agent_hybrid.py -- V2 Planner + Semantic Router Browser Agent

Architecture:
    1. Pre-Flight Planner (LLM Call 1): Generates a step-by-step plan from the macro-goal
    2. Semantic Router (Embedding Layer): Uses nomic-embed-text to filter DOM elements
       via cosine similarity, keeping only the Top-N most relevant elements
    3. Micro-Executor (LLM Call 2): Lightweight LLM that executes one sub-step at a time
    4. Python State Machine: Orchestrates the loop and advances through the plan

Flow per step:
    Extract DOM → Semantic Filter → Micro-Executor decides → Playwright acts
    When executor says "done", advance to next sub-step.
    When all sub-steps complete, mission accomplished.
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
import argparse
from PIL import Image
from playwright.sync_api import sync_playwright

os.environ["PYTHONUNBUFFERED"] = "1"

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_EMBED_ENDPOINT = "http://localhost:11434/api/embed"
DOM_MODEL = "qwen3-vl:8b-instruct"      # Used for planning + micro-execution. Prefix with "hf:" for HuggingFace (e.g. "hf:Paulescu/LFM2-350M-browsergym-20251224-013119")
VISION_MODEL = "qwen3-vl:8b-instruct"   # Vision fallback (coordinate output) — always Ollama, must support images
EMBED_MODEL = "nomic-embed-text"         # Local embedding model for semantic routing

# Semantic router config
TOP_N_ELEMENTS = 7                       # Max elements to pass to micro-executor
SIMILARITY_THRESHOLD = 0.25              # Minimum cosine similarity to include

# Regex for structural elements that always pass through the semantic filter.
# Matches the actual DOM extraction format: "input[text]", "input[search]",
# "select:", "textarea", and role-based elements like [role="combobox"].
STRUCTURAL_BYPASS_PATTERN = re.compile(
    r'\binput\[|'               # input[text], input[search], etc.
    r'\btextarea\b|'            # textarea elements
    r'\bselect\b|'              # select (dropdown) elements
    r'role="(combobox|searchbox|textbox|listbox|menu|dialog)"',
    re.IGNORECASE
)


def _print(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------
# Model Backend Abstraction
# Ollama: default, supports JSON schema enforcement
# HuggingFace: prefix model name with "hf:" to use
#   e.g. DOM_MODEL = "hf:Paulescu/LFM2-350M-browsergym-20251224-013119"
# Note: Vision fallback always uses Ollama — text-only HF models cannot see images
# ---------------------------------------------------------------

def _extract_json_from_text(text):
    """Extract JSON object or array from raw LLM output. Used for HF models that can't enforce schema."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try extract first JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    # Try extract first JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


class OllamaBackend:
    """Wraps Ollama /api/generate endpoint. Supports JSON schema enforcement."""

    def __init__(self, model_id):
        self.model_id = model_id

    def generate(self, prompt, schema=None):
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        if schema:
            payload["format"] = schema
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        return response.json().get("response", "")


class HuggingFaceBackend:
    """Wraps a HuggingFace text model via the transformers library.
    Requires: pip install transformers torch accelerate
    No image support — DOM/text tasks only. Vision fallback stays on Ollama.
    """

    def __init__(self, model_id):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "HuggingFace backend requires: pip install transformers torch accelerate"
            )
        _print(f"[HF] Loading model '{model_id}' — this may take a minute on first run ...")
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        device = next(self.model.parameters()).device
        _print(f"[HF] Model loaded on {device}.")

    def generate(self, prompt, schema=None):
        # schema is ignored — HF models can't enforce it. JSON hint is added to prompt instead.
        messages = [{"role": "user", "content": prompt}]

        # apply_chat_template → string first, then tokenize separately.
        # Avoids BatchEncoding vs tensor confusion across transformers versions.
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(formatted_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with self._torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                    if self.tokenizer.eos_token_id is not None
                    else self.tokenizer.pad_token_id
                )
            )
        new_tokens = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# Singleton — model is loaded once and reused across all calls
_dom_backend_instance = None

def get_dom_backend():
    global _dom_backend_instance
    if _dom_backend_instance is None:
        if DOM_MODEL.startswith("hf:"):
            model_id = DOM_MODEL[3:]
            _dom_backend_instance = HuggingFaceBackend(model_id)
        else:
            _dom_backend_instance = OllamaBackend(DOM_MODEL)
    return _dom_backend_instance


# ---------------------------------------------------------------
# 1. DOM Extraction
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
                } else if (tag === 'select') {
                    // For select: show the aria-label or the currently selected option text
                    let label = node.getAttribute('aria-label') || '';
                    let selectedText = node.options && node.selectedIndex >= 0
                        ? node.options[node.selectedIndex].text : '';
                    text = label ? label + ' (current: ' + selectedText + ')' : selectedText;
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
# 3. OpenCV Sniper (for vision fallback)
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
# 5. PRE-FLIGHT PLANNER (Phase 2) — LLM Call 1
#    Runs once before Playwright launches. Decomposes the macro-goal
#    into a strict JSON array of chronological sub-steps.
# ---------------------------------------------------------------

def generate_plan(goal):
    """Generate a step-by-step plan from the macro-goal. Returns a list of strings."""
    _print(f"\n[PLANNER] Generating plan for: {goal}")

    prompt = f"""You are a web navigation planner. Your job is to break down a user's goal into a precise, chronological list of browser actions.

GOAL: {goal}

Rules:
- Output ONLY a JSON array of strings. No explanation, no markdown.
- Each step must be a COMPLETE, self-contained browser action.
- IMPORTANT: Merge related sub-actions into one step. For example:
  - WRONG: ["Click the search box", "Type 'protein powder'", "Press Enter"] (3 steps!)
  - RIGHT: ["Search for 'protein powder' in the search box"] (1 step — the system handles click, type, and Enter)
  - WRONG: ["Click the sort dropdown", "Select 'Price: Low to High'"] (2 steps!)
  - RIGHT: ["Sort results by 'Price: Low to High'"] (1 step)
- Include navigation steps (e.g., "Go to amazon.in").
- The final step should describe the completion state (e.g., "Confirm the cheapest item is visible on screen").
- Keep steps concrete and unambiguous. Use the exact text/labels the user would see.
- Typically 3-7 steps. Be thorough but not redundant.

Example output for "Search Amazon for protein powder sorted by lowest price":
["Go to amazon.in", "Search for 'protein powder' in the search box", "Sort results by 'Price: Low to High'", "Click the cheapest actual protein supplement product"]

Output the JSON array now:"""

    schema = {
        "type": "array",
        "items": {"type": "string"}
    }

    # For HF models that can't enforce schema, add an explicit JSON hint to the prompt
    backend = get_dom_backend()
    if isinstance(backend, HuggingFaceBackend):
        prompt += "\n\nIMPORTANT: Output ONLY a valid JSON array of strings. No explanation, no markdown, no code fences. Example: [\"step 1\", \"step 2\"]"

    try:
        text_output = backend.generate(prompt, schema=schema)
        plan = _extract_json_from_text(text_output) if isinstance(backend, HuggingFaceBackend) else json.loads(text_output)
        if isinstance(plan, list) and len(plan) > 0:
            _print(f"[PLANNER] Generated {len(plan)} steps:")
            for i, step_text in enumerate(plan):
                _print(f"  {i+1}. {step_text}")
            return plan
        else:
            _print("[PLANNER] Invalid plan format, using fallback single-step plan.")
            return [goal]
    except Exception as e:
        _print(f"[PLANNER] Plan generation failed: {e}")
        _print("[PLANNER] Falling back to single-step plan.")
        return [goal]


# ---------------------------------------------------------------
# 6. SEMANTIC ROUTER (Phase 3) — The Math Layer
#    Uses nomic-embed-text to filter DOM elements by relevance
#    to the current sub-step via cosine similarity.
# ---------------------------------------------------------------

def _get_embeddings(texts):
    """Get embeddings for a list of texts from nomic-embed-text via Ollama."""
    payload = {
        "model": EMBED_MODEL,
        "input": texts
    }
    response = requests.post(OLLAMA_EMBED_ENDPOINT, json=payload)
    data = response.json()
    return data.get("embeddings", [])


def _cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two numpy vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_filter_dom(current_step, dom_elements_str):
    """
    Filter DOM elements by semantic relevance to the current sub-step.

    Args:
        current_step: The current plan step string (e.g., "Search for 'protein powder'")
        dom_elements_str: Raw string of DOM elements from extract_viewport_elements()

    Returns:
        Filtered string of the most relevant DOM elements (Top-N + structural bypasses)
    """
    if not dom_elements_str or not dom_elements_str.strip():
        return dom_elements_str

    lines = [line.strip() for line in dom_elements_str.strip().split('\n') if line.strip()]

    if len(lines) <= TOP_N_ELEMENTS:
        _print(f"  [ROUTER] Only {len(lines)} elements, skipping filter (all pass).")
        return dom_elements_str

    # Separate structural bypasses from candidates
    bypass_lines = []
    candidate_lines = []

    for line in lines:
        if STRUCTURAL_BYPASS_PATTERN.search(line):
            bypass_lines.append(line)
        else:
            candidate_lines.append(line)

    _print(f"  [ROUTER] {len(bypass_lines)} structural bypasses, {len(candidate_lines)} candidates to rank.")

    if not candidate_lines:
        return '\n'.join(bypass_lines)

    # Get embeddings: step + all candidate element texts
    texts_to_embed = [current_step] + candidate_lines

    try:
        embeddings = _get_embeddings(texts_to_embed)
    except Exception as e:
        _print(f"  [ROUTER] Embedding failed: {e}. Passing all elements through.")
        return dom_elements_str

    if len(embeddings) < len(texts_to_embed):
        _print(f"  [ROUTER] Got {len(embeddings)} embeddings for {len(texts_to_embed)} texts. Passing all.")
        return dom_elements_str

    step_embedding = embeddings[0]
    element_embeddings = embeddings[1:]

    # Calculate cosine similarity for each candidate
    scored = []
    for i, elem_emb in enumerate(element_embeddings):
        sim = _cosine_similarity(step_embedding, elem_emb)
        scored.append((sim, candidate_lines[i]))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top N that pass the threshold
    slots_remaining = TOP_N_ELEMENTS - len(bypass_lines)
    top_candidates = []
    for sim, line in scored:
        if slots_remaining <= 0:
            break
        if sim >= SIMILARITY_THRESHOLD:
            top_candidates.append(line)
            _print(f"    [ROUTER] {sim:.3f} | {line[:80]}")
            slots_remaining -= 1

    # If we got nothing above threshold, take at least top 3
    if not top_candidates and scored:
        for sim, line in scored[:3]:
            top_candidates.append(line)
            _print(f"    [ROUTER] (below threshold) {sim:.3f} | {line[:80]}")

    # Combine: bypasses first, then ranked candidates
    final_elements = bypass_lines + top_candidates
    _print(f"  [ROUTER] Final: {len(final_elements)} elements ({len(bypass_lines)} bypass + {len(top_candidates)} ranked)")

    return '\n'.join(final_elements)


# ---------------------------------------------------------------
# 7. MICRO-EXECUTOR (Phase 4) — LLM Call 2
#    Lightweight, focused on executing a single sub-step.
#    Receives only the filtered DOM elements from the semantic router.
# ---------------------------------------------------------------

def get_micro_decision(filtered_dom, goal, current_step, step_index, total_steps,
                       history, current_url, page_title):
    """Lightweight executor: decides the next action for the current sub-step."""
    history_str = json.dumps(history[-6:], indent=2) if history else "[]"

    prompt = f"""You are a web UI executor. You execute ONE sub-step at a time.

MACRO GOAL: {goal}
CURRENT SUB-STEP ({step_index + 1}/{total_steps}): {current_step}

CURRENT STATE:
- URL: {current_url}
- Page Title: {page_title}

PAST ACTIONS (recent):
{history_str}

Here are the {len(filtered_dom.splitlines()) if filtered_dom else 0} relevant UI elements:
{filtered_dom}

Your job: Output a JSON action to achieve THIS SPECIFIC sub-step (not the entire goal).

Available actions:
- "click": Click an element by element_id.
- "type": Type text into an input field by element_id. The system will CLEAR the field first, then type, then press Enter.
- "select_option": Select a dropdown option. Provide element_id of the <select> and the "text" of the option to choose (e.g., "Price: Low to High").
- "scroll": Scroll down to reveal more content.
- "goto": Navigate to a URL (use for going to a new website).
- "done": This sub-step is complete. Use when the current sub-step's objective has been achieved.
- "vision_fallback": The target is visual (image/thumbnail) and not in the DOM list. Describe what to look for.

Critical rules:
- "done" means THIS SUB-STEP is finished, not the entire macro goal.
- If the sub-step says "Go to X" and the URL already shows X, output "done".
- If the sub-step says "Search for X": find the input field, use "type" with element_id and text. The system handles clicking, clearing, typing, and pressing Enter.
- If the sub-step says "Sort by X" and there's a <select> dropdown: use "select_option" with the select's element_id and the option text.
- Do NOT repeat failed actions. Try something different.
- "click" and "type" ALWAYS require an element_id (integer). Never output them with null element_id.
- Keep reasoning under 2 sentences.
- NEVER use "type" to enter a URL. Use "goto" with the full URL instead.
"""

    schema = {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "1-2 sentence explanation."
            },
            "action": {
                "type": "string",
                "enum": ["click", "type", "scroll", "done", "vision_fallback", "goto", "select_option"],
                "description": "The action to take for this sub-step."
            },
            "url": {
                "type": ["string", "null"],
                "description": "Full URL for 'goto' action. null otherwise."
            },
            "element_id": {
                "type": ["integer", "null"],
                "description": "ID of element for click/type/select_option. null for scroll/done/vision_fallback/goto."
            },
            "text": {
                "type": ["string", "null"],
                "description": "Text for type/select_option actions. For type: the text to enter. For select_option: the visible option text to select. null otherwise."
            },
            "vision_description": {
                "type": ["string", "null"],
                "description": "For vision_fallback: describe the visual element to find. null otherwise."
            }
        },
        "required": ["reasoning", "action"]
    }

    backend = get_dom_backend()
    if isinstance(backend, HuggingFaceBackend):
        prompt += "\n\nIMPORTANT: Output ONLY a valid JSON object. No explanation, no markdown, no code fences."

    _print("  [EXEC-LLM] Thinking ...")
    try:
        text_output = backend.generate(prompt, schema=schema)
        result = _extract_json_from_text(text_output) if isinstance(backend, HuggingFaceBackend) else json.loads(text_output)
        if result is None:
            raise ValueError("No JSON found in output")
        _print(f"  [EXEC-LLM] Reasoning: {result.get('reasoning', '')}")
        _print(f"  [EXEC-LLM] Action: {result.get('action')} | ID: {result.get('element_id')} | Text: {result.get('text', '')}")
        return result
    except Exception as e:
        _print(f"  [EXEC-LLM] Parse error: {e}")
        return None


# ---------------------------------------------------------------
# 8. Vision Fallback — Full Decision-Maker
# ---------------------------------------------------------------

def get_vision_decision(base64_image, goal, history, current_url, page_title, hint=""):
    """Vision LLM call. Uses screenshot for click/type/scroll/done."""
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
# 9. THE STATE MACHINE (Phase 5) — The Hybrid Loop V2
#    Plan → For each sub-step: Extract DOM → Semantic Filter →
#    Micro-Executor → Playwright acts → Advance on "done"
# ---------------------------------------------------------------

def _is_select_element(page, element_id):
    """Check if an element with data-ai-id is a <select> tag."""
    try:
        tag = page.locator(f"[data-ai-id='{element_id}']").evaluate("el => el.tagName.toLowerCase()")
        return tag == "select"
    except Exception:
        return False


def run_hybrid_agent(start_url, goal, max_steps=25):
    _print(f"\n{'='*50}")
    _print(f" V2 PLANNER + SEMANTIC ROUTER AGENT")
    _print(f" GOAL: '{goal}'")
    _print(f"{'='*50}\n")

    # ---- Phase 2: Pre-Flight Planning (before browser launches) ----
    plan = generate_plan(goal)
    current_step_index = 0

    _print(f"\n[STATE MACHINE] Plan has {len(plan)} steps. Launching browser...\n")

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
        substep_attempts = 0      # Track attempts per sub-step to avoid infinite loops
        MAX_SUBSTEP_ATTEMPTS = 8  # Max actions per sub-step before forcing advance

        try:
            while step <= max_steps:
                # Check if we've completed all plan steps
                if current_step_index >= len(plan):
                    _print(f"\n  [SUCCESS] All {len(plan)} plan steps completed!")
                    _print(f"  Final URL: {page.url}")
                    _print(f"  Final Title: {page.title()}")
                    time.sleep(5)
                    break

                current_step = plan[current_step_index]
                _print(f"\n--- Step {step}/{max_steps} | Plan [{current_step_index + 1}/{len(plan)}]: {current_step} ---")

                # -- State snapshot --
                page.wait_for_load_state("domcontentloaded")
                time.sleep(1.5)
                current_url = page.url
                page_title = page.title()
                _print(f"  [STATE] URL: {current_url}")
                _print(f"  [STATE] Title: {page_title}")

                # -- DOM extraction --
                dom_elements = extract_viewport_elements(page)

                # -- Phase 3: Semantic Router --
                filtered_dom = semantic_filter_dom(current_step, dom_elements)

                # -- Phase 4: Micro-Executor --
                decision = get_micro_decision(
                    filtered_dom, goal, current_step,
                    current_step_index, len(plan),
                    action_history, current_url, page_title
                )

                if not decision:
                    _print("  [WARN] Decision failed, retrying ...")
                    action_history.append({"step": step, "action": "error", "status": "FAILED"})
                    step += 1
                    substep_attempts += 1
                    continue

                action = decision.get("action")
                element_id = decision.get("element_id")

                # ============================================
                # DONE — Sub-step complete, advance the plan
                # ============================================
                if action == "done":
                    _print(f"\n  [SUCCESS] Step complete: '{current_step}'")
                    current_step_index += 1
                    substep_attempts = 0
                    action_history.append({
                        "step": step, "action": "substep_done",
                        "completed_step": current_step,
                        "status": "success"
                    })

                    # Check if that was the last step
                    if current_step_index >= len(plan):
                        _print(f"\n  [MISSION ACCOMPLISHED] All {len(plan)} steps completed!")
                        _print(f"  URL: {current_url}")
                        _print(f"  Title: {page_title}")
                        time.sleep(5)
                        break
                    else:
                        _print(f"  [NEXT] Step {current_step_index + 1}/{len(plan)}: {plan[current_step_index]}")
                    step += 1
                    continue

                # ============================================
                # GOTO — Virtual Address Bar
                # ============================================
                elif action == "goto":
                    target_url = decision.get("url")
                    if target_url:
                        if not target_url.startswith("http"):
                            target_url = "https://" + target_url
                        _print(f"  [EXEC] Navigating to: {target_url}")
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
                        _print("  [EXEC] Goto failed: No URL provided.")

                # ============================================
                # SELECT_OPTION — Native dropdown selection
                # Uses page.select_option() to bypass overlay issues
                # ============================================
                elif action == "select_option" and element_id:
                    option_text = decision.get("text", "")
                    try:
                        selector = f"[data-ai-id='{element_id}']"
                        page.select_option(selector, label=option_text)
                        _print(f"  [EXEC] Selected '{option_text}' from dropdown #{element_id}")
                        time.sleep(2.0)  # Wait for page to reload with new sort
                        action_history.append({
                            "step": step, "action": "select_option",
                            "id": element_id, "text": option_text,
                            "status": "success", "url_after": page.url
                        })
                    except Exception as e:
                        _print(f"  [EXEC] select_option failed: {e}")
                        # Fallback: try JS-based selection
                        try:
                            page.evaluate(f"""() => {{
                                const sel = document.querySelector("[data-ai-id='{element_id}']");
                                if (sel) {{
                                    for (let opt of sel.options) {{
                                        if (opt.text.includes("{option_text}")) {{
                                            sel.value = opt.value;
                                            sel.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                            break;
                                        }}
                                    }}
                                }}
                            }}""")
                            _print(f"  [EXEC] JS fallback: selected '{option_text}' from #{element_id}")
                            time.sleep(2.0)
                            action_history.append({
                                "step": step, "action": "select_option_js",
                                "id": element_id, "text": option_text,
                                "status": "success", "url_after": page.url
                            })
                        except Exception as e2:
                            _print(f"  [EXEC] JS fallback also failed: {e2}")
                            action_history.append({
                                "step": step, "action": "select_option",
                                "id": element_id, "text": option_text,
                                "status": "FAILED", "error": str(e2)
                            })

                # ============================================
                # CLICK — DOM native via element ID
                # If the element is a <select>, auto-convert to select_option
                # ============================================
                elif action == "click" and element_id:
                    # Auto-detect if this is a <select> and convert to select_option
                    if _is_select_element(page, element_id):
                        _print(f"  [EXEC] Element #{element_id} is a <select>. Use 'select_option' action instead.")
                        _print(f"  [EXEC] Hint: Tell the LLM to use select_option with the option text.")
                        action_history.append({
                            "step": step, "action": "click",
                            "id": element_id, "status": "FAILED",
                            "error": "Element is a <select> dropdown. Use select_option action with the desired option text instead of click."
                        })
                    else:
                        try:
                            locator = page.locator(f"[data-ai-id='{element_id}']")
                            locator.click(timeout=3000, force=True)
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
                # TYPE — Clear field first, then type + Enter
                # ============================================
                elif action == "type" and element_id:
                    text_to_type = decision.get("text", "")
                    try:
                        locator = page.locator(f"[data-ai-id='{element_id}']")
                        # Click to focus
                        locator.click(timeout=3000)
                        time.sleep(0.3)
                        # Select all existing text and clear it before typing
                        page.keyboard.press("Control+a")
                        time.sleep(0.1)
                        page.keyboard.press("Backspace")
                        time.sleep(0.2)
                        # Now type the new text
                        page.keyboard.type(text_to_type, delay=50)
                        time.sleep(0.5)
                        page.keyboard.press("Enter")
                        _print(f"  [EXEC] Typed '{text_to_type}' into #{element_id} (cleared first) + Enter")
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
                # TYPE without element_id — press Enter or key
                # ============================================
                elif action == "type" and not element_id:
                    text_to_type = decision.get("text", "")
                    if text_to_type:
                        _print(f"  [EXEC] Typing '{text_to_type}' into focused element + Enter")
                        page.keyboard.type(text_to_type, delay=50)
                        time.sleep(0.5)
                        page.keyboard.press("Enter")
                    else:
                        _print(f"  [EXEC] Pressing Enter (type with no element/text)")
                        page.keyboard.press("Enter")
                    action_history.append({
                        "step": step, "action": "key_press",
                        "text": text_to_type or "Enter", "status": "success"
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

                    if v_action == "done":
                        _print(f"\n  [SUCCESS] Step complete (via vision): '{current_step}'")
                        current_step_index += 1
                        substep_attempts = 0
                        action_history.append({
                            "step": step, "action": "vision_substep_done",
                            "completed_step": current_step, "status": "success"
                        })
                        if current_step_index >= len(plan):
                            _print(f"\n  [MISSION ACCOMPLISHED] All steps done! (via vision)")
                            time.sleep(5)
                            break
                        else:
                            _print(f"  [NEXT] Step {current_step_index + 1}/{len(plan)}: {plan[current_step_index]}")
                        step += 1
                        continue

                    elif v_action == "scroll":
                        _print("  [VISION] Scrolling down ...")
                        page.keyboard.press("PageDown")
                        time.sleep(2.0)
                        action_history.append({
                            "step": step, "action": "vision_scroll",
                            "url": current_url, "status": "success"
                        })

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
                            # Clear before typing in vision mode too
                            page.keyboard.press("Control+a")
                            time.sleep(0.1)
                            page.keyboard.press("Backspace")
                            time.sleep(0.2)
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

                # Track sub-step attempts to avoid infinite loops
                substep_attempts += 1
                if substep_attempts >= MAX_SUBSTEP_ATTEMPTS:
                    _print(f"  [WARN] Max attempts ({MAX_SUBSTEP_ATTEMPTS}) for sub-step '{current_step}'. Force-advancing.")
                    current_step_index += 1
                    substep_attempts = 0

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
    parser = argparse.ArgumentParser(description="V2 Planner + Semantic Router Browser Agent")
    parser.add_argument("--goal", required=True, help="Task for the agent to complete")
    parser.add_argument("--start-url", default="about:blank", help="URL to open on launch")
    parser.add_argument("--max-steps", type=int, default=25, help="Maximum agent steps")
    parser.add_argument(
        "--dom-model", default=None,
        help="Model for planning + execution. Use 'hf:<model_id>' for HuggingFace "
             "(e.g. 'hf:Paulescu/LFM2-350M-browsergym-20251224-013119'). "
             "Default: uses DOM_MODEL from config."
    )
    parser.add_argument(
        "--vision-model", default=None,
        help="Ollama model for vision fallback. Default: uses VISION_MODEL from config."
    )
    args = parser.parse_args()

    # Override globals if passed via CLI
    if args.dom_model:
        DOM_MODEL = args.dom_model
    if args.vision_model:
        VISION_MODEL = args.vision_model

    run_hybrid_agent(
        start_url=args.start_url,
        goal=args.goal,
        max_steps=args.max_steps,
    )
