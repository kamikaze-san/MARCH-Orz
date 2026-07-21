"""
agent_lfm2.py — Minimal BrowserGym-style loop for LFM2-350M

No planning layer. No JSON schema. No scratchpad.
Just: observe (AXTree) → model reasons → parse action → execute → repeat

The model was trained on BrowserGym observations (AXTree + bid IDs).
We feed it exactly that format and parse its natural output.

Run:
    python agent_lfm2.py --goal "Search Google for python tutorials"
    python agent_lfm2.py --goal "..." --start-url "https://google.com"
    python agent_lfm2.py --goal "..." --model "some-other/hf-model"
"""

import argparse
import re
import time
import os
from playwright.sync_api import sync_playwright

os.environ["PYTHONUNBUFFERED"] = "1"

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "Paulescu/LFM2-350M-browsergym-20251224-013119"
MAX_NEW_TOKENS = 512

# Roles we assign a bid to (interactive elements the model can act on)
INTERACTIVE_ROLES = {
    "button", "link", "combobox", "textbox", "searchbox",
    "checkbox", "radio", "menuitem", "option", "tab",
    "slider", "spinbutton", "listbox", "menuitemcheckbox",
    "menuitemradio", "switch", "treeitem",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _print(msg):
    print(msg, flush=True)


# ── 1. Model loading ──────────────────────────────────────────────────────────
def load_model(model_id: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("Run: pip install transformers torch accelerate")

    _print(f"[MODEL] Loading '{model_id}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    _print(f"[MODEL] Loaded on {device}\n")
    return model, tokenizer


def call_model(model, tokenizer, prompt: str) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_ids     = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=(
                tokenizer.eos_token_id
                if tokenizer.eos_token_id is not None
                else tokenizer.pad_token_id
            ),
        )

    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── 2. AXTree extraction ──────────────────────────────────────────────────────
def extract_axtree(page):
    """
    Get Playwright accessibility snapshot → flat BrowserGym-style text + bid map.

    Returns:
        axtree_text : string shown to the model
        bid_map     : {bid_int: (role, name)} for locating elements later
    """
    snapshot = page.accessibility.snapshot(interesting_only=True)
    if not snapshot:
        return "(page has no accessible content)", {}

    lines   = []
    bid_map = {}
    counter = [0]

    def walk(node, depth=0):
        if not isinstance(node, dict):
            return

        role  = node.get("role", "")
        name  = node.get("name", "") or ""
        value = node.get("value", "") or ""

        # skip invisible/structural noise
        if not role or role in ("none", "presentation", "generic", "InlineTextBox"):
            for child in node.get("children", []):
                walk(child, depth)
            return

        indent = "  " * depth

        if role in INTERACTIVE_ROLES:
            counter[0] += 1
            bid = counter[0]
            bid_map[bid] = (role, name)
            # Match training format exactly: [13] button 'Click Me!'
            lines.append(f"{indent}[{bid}] {role} '{name}'")
        else:
            # informational node — show but no bid
            display = f"{role} '{name}'" if name else role
            lines.append(indent + display)

        for child in node.get("children", []):
            walk(child, depth + 1)

    walk(snapshot)
    return "\n".join(lines) if lines else "(no interactive elements found)", bid_map


# ── 3. Prompt builder ─────────────────────────────────────────────────────────
def build_prompt(goal: str, url: str, title: str, axtree_text: str, history: list) -> str:
    # Minimal format matching training data:
    # goal on top, raw AXTree below, last action if any — nothing else.
    last_action = f"\nLast action: {history[-1]}" if history else ""
    return f"""Goal: {goal}
URL: {url}
{axtree_text}{last_action}"""


# ── 4. Action parser ──────────────────────────────────────────────────────────
def parse_action(text: str):
    """
    Find the LAST valid action in the model output (model reasons first, acts last).
    Returns a dict or None.
    """
    patterns = [
        # type(bid, "text") — check before click so it matches first
        (r'type\s*\(\s*["\']?(\d+)["\']?\s*,\s*["\']([^"\']*)["\']',   "type"),
        # click(bid)
        (r'click\s*\(\s*["\']?(\d+)["\']?\s*\)',                         "click"),
        # goto("url")
        (r'goto\s*\(\s*["\']([^"\']+)["\']\s*\)',                        "goto"),
        # scroll(direction)
        (r'scroll\s*\(\s*["\']?(up|down)["\']?\s*\)',                    "scroll"),
        # stop("answer")
        (r'stop\s*\(\s*["\']([^"\']*)["\']?\s*\)',                       "stop"),
    ]

    best_pos   = -1
    best_result = None

    for pattern, action_type in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if m.start() > best_pos:
                best_pos = m.start()
                if action_type == "click":
                    best_result = {"action": "click", "bid": int(m.group(1))}
                elif action_type == "type":
                    best_result = {"action": "type", "bid": int(m.group(1)), "text": m.group(2)}
                elif action_type == "goto":
                    url = m.group(1)
                    if not url.startswith("http"):
                        url = "https://" + url
                    best_result = {"action": "goto", "url": url}
                elif action_type == "scroll":
                    best_result = {"action": "scroll", "direction": m.group(1).lower()}
                elif action_type == "stop":
                    best_result = {"action": "stop", "answer": m.group(1)}

    return best_result


# ── 5. Action executor ────────────────────────────────────────────────────────
def execute_action(page, action: dict, bid_map: dict) -> str:
    """Execute parsed action. Returns a one-line outcome description."""
    act = action["action"]

    if act == "goto":
        url = action["url"]
        _print(f"  [EXEC] goto → {url}")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return f"navigated to {url}"
        except Exception as e:
            return f"goto failed: {e}"

    elif act == "scroll":
        direction = action.get("direction", "down")
        _print(f"  [EXEC] scroll {direction}")
        page.keyboard.press("PageDown" if direction == "down" else "PageUp")
        time.sleep(1.5)
        return f"scrolled {direction}"

    elif act == "stop":
        return "__STOP__"

    # click and type both need a bid → locator
    bid = action.get("bid")
    if bid is None or bid not in bid_map:
        return f"bid {bid} not found in current AXTree"

    role, name = bid_map[bid]
    _print(f"  [EXEC] {act} → [{bid}] {role} '{name}'")

    try:
        # Build locator: prefer get_by_role with name, fall back to get_by_label
        if name:
            locator = page.get_by_role(role, name=name).first
        else:
            locator = page.get_by_role(role).first

        if act == "click":
            locator.click(timeout=8000)
            time.sleep(1.5)
            return f"clicked [{bid}] {role} '{name}'"

        elif act == "type":
            text = action.get("text", "")
            locator.click(timeout=8000)
            locator.fill(text)
            page.keyboard.press("Enter")
            time.sleep(2.0)
            return f"typed '{text}' into [{bid}] {role} '{name}'"

    except Exception as e:
        _print(f"  [EXEC] Locator failed ({e}), trying get_by_label fallback ...")
        try:
            if act == "click":
                page.get_by_label(name).first.click(timeout=5000)
                time.sleep(1.5)
                return f"clicked (via label) '{name}'"
            elif act == "type":
                text = action.get("text", "")
                loc = page.get_by_label(name).first
                loc.click(timeout=5000)
                loc.fill(text)
                page.keyboard.press("Enter")
                time.sleep(2.0)
                return f"typed '{text}' (via label) '{name}'"
        except Exception as e2:
            return f"execution failed: role={role} name='{name}' | {e2}"

    return "unknown execution path"


# ── 6. Main agent loop ────────────────────────────────────────────────────────
def run_agent(goal: str, start_url: str, model_id: str, max_steps: int = 20):
    _print(f"\n{'='*55}")
    _print(f"  LFM2 BROWSER AGENT")
    _print(f"  Goal:  {goal}")
    _print(f"  Model: {model_id}")
    _print(f"{'='*55}\n")

    model, tokenizer = load_model(model_id)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--window-size=1280,900", "--disable-infobars"]
        )
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page    = context.new_page()

        if start_url and start_url != "about:blank":
            _print(f"[NAV] Opening {start_url} ...")
            page.goto(start_url, wait_until="domcontentloaded")

        history = []

        try:
            for step in range(1, max_steps + 1):
                page.wait_for_load_state("domcontentloaded")
                time.sleep(1.0)

                url   = page.url
                title = page.title()
                _print(f"\n--- Step {step}/{max_steps} ---")
                _print(f"  URL:   {url}")
                _print(f"  Title: {title}")

                # Observe
                axtree_text, bid_map = extract_axtree(page)
                _print(f"  AXTree: {len(bid_map)} interactive elements")

                # Think
                prompt = build_prompt(goal, url, title, axtree_text, history)
                _print("  [LLM] Thinking ...")
                raw_output = call_model(model, tokenizer, prompt)
                _print(f"  [LLM] Output:\n{raw_output}\n")

                # Parse
                action = parse_action(raw_output)
                if action is None:
                    _print("  [WARN] No valid action found in output. Scrolling as fallback.")
                    action = {"action": "scroll", "direction": "down"}

                _print(f"  [ACTION] {action}")

                # Execute
                if action["action"] == "stop":
                    answer = action.get("answer", "")
                    _print(f"\n  ✅ DONE — Model says: {answer or '(no answer given)'}")
                    break

                outcome = execute_action(page, action, bid_map)
                if outcome == "__STOP__":
                    _print(f"\n  ✅ DONE")
                    break

                _print(f"  [OUTCOME] {outcome}")
                history.append(f"step {step}: {action['action']} → {outcome}")

            else:
                _print(f"\n  [MAX STEPS] Reached {max_steps} steps without stopping.")

        finally:
            time.sleep(3)
            browser.close()
            _print("  Browser closed.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFM2 BrowserGym-style agent")
    parser.add_argument("--goal",      required=True,                    help="Task for the agent")
    parser.add_argument("--start-url", default="about:blank",            help="Starting URL")
    parser.add_argument("--model",     default=DEFAULT_MODEL,            help="HuggingFace model ID")
    parser.add_argument("--max-steps", type=int, default=20,             help="Max loop iterations")
    args = parser.parse_args()

    run_agent(
        goal      = args.goal,
        start_url = args.start_url,
        model_id  = args.model,
        max_steps = args.max_steps,
    )
