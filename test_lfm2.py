"""
test_lfm2.py — Probe script for Paulescu/LFM2-350M-browsergym

Run:  python test_lfm2.py
      python test_lfm2.py --model "hf:some-other-model"

What this does:
  1. Loads the model via HuggingFace transformers
  2. Runs 4 probe prompts in order:
       A) Sanity check  — plain language, verify model is alive
       B) Our format    — our current JSON schema prompt, see if it can follow it
       C) BrowserGym format — AXTree + bid IDs, the format it was trained on
       D) Minimal action  — "given this page, what do you click?" stripped to minimum
  3. Prints raw output for each so you can see exactly what format it prefers

Goal: understand what prompt → what output before trying to integrate into agent_hybrid.py
"""

import argparse
import json
import re

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "Paulescu/LFM2-350M-browsergym-20251224-013119"

# Fake minimal AXTree snapshot (BrowserGym format, with bid IDs)
FAKE_AXTREE = """
[1] RootWebArea 'Google'
  [2] navigation ''
    [3] link 'About'
    [4] link 'Store'
  [5] main ''
    [6] heading 'Google'
    [7] combobox 'Search' bid='search-box'
    [8] button 'Google Search' bid='search-btn'
    [9] button "I'm Feeling Lucky" bid='lucky-btn'
""".strip()

# ── Probe prompts ────────────────────────────────────────────────────────────
PROBES = [
    {
        "name": "A — Sanity check (plain language)",
        "prompt": "You are a helpful assistant. What is 2 + 2? Answer in one word.",
    },
    {
        "name": "B — Our current JSON schema format",
        "prompt": """You are a web UI executor.

MACRO GOAL: Search Google for 'python tutorials'
CURRENT SUB-STEP (1/2): Type 'python tutorials' into the search box and press Enter

CURRENT STATE:
- URL: https://www.google.com
- Page Title: Google

UI ELEMENTS:
1 | input[search] | placeholder='Search'
2 | button | text='Google Search'
3 | button | text="I'm Feeling Lucky"

Output a JSON object:
{
  "reasoning": "1-2 sentence explanation",
  "action": "click" | "type" | "scroll" | "goto" | "done",
  "element_id": <integer or null>,
  "text": "<text to type or null>"
}"""
    },
    {
        "name": "C — BrowserGym format (what it was trained on)",
        "prompt": f"""You are a web agent. Your goal is: Search Google for 'python tutorials'

Current page observation:
URL: https://www.google.com
Title: Google

Accessibility Tree:
{FAKE_AXTREE}

Available actions:
- click(bid)            : click element with browser ID
- type(bid, text)       : type text into element
- scroll(direction)     : scroll 'up' or 'down'
- goto(url)             : navigate to URL
- stop(answer)          : stop and return answer

What is your next action? Think step by step then output the action."""
    },
    {
        "name": "D — Minimal stripped prompt",
        "prompt": """Page: Google search homepage
Goal: search for 'python tutorials'
Elements: search-box (text input), search-btn (button)
Action:"""
    },
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_model(model_id: str):
    print(f"\n[LOAD] Loading '{model_id}' ...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("Run: pip install transformers torch accelerate")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[LOAD] Done. Device: {device}\n")
    return model, tokenizer


def run_probe(model, tokenizer, probe: dict) -> str:
    import torch

    messages = [{"role": "user", "content": probe["prompt"]}]

    # apply_chat_template → formatted string first, then tokenize separately.
    # Avoids BatchEncoding vs tensor confusion across transformers versions.
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=(
                tokenizer.eos_token_id
                if tokenizer.eos_token_id is not None
                else tokenizer.pad_token_id
            ),
        )

    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def print_section(title: str, content: str):
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(content)
    print(bar)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Probe LFM2 browsergym model")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace model ID (no 'hf:' prefix needed here)"
    )
    parser.add_argument(
        "--probe", type=int, default=None,
        help="Run only probe N (0=A, 1=B, 2=C, 3=D). Default: run all."
    )
    args = parser.parse_args()

    model_id = args.model.replace("hf:", "")  # strip prefix if passed with it
    model, tokenizer = load_model(model_id)

    probes_to_run = [PROBES[args.probe]] if args.probe is not None else PROBES

    results = {}
    for probe in probes_to_run:
        print(f"\n▶ Running probe: {probe['name']}")
        print_section("PROMPT", probe["prompt"])
        output = run_probe(model, tokenizer, probe)
        print_section("RAW OUTPUT", output)
        results[probe["name"]] = output

    # Summary
    print("\n" + "═" * 60)
    print("  SUMMARY — Raw outputs per probe")
    print("═" * 60)
    for name, out in results.items():
        print(f"\n[{name}]")
        print(out[:300] + ("..." if len(out) > 300 else ""))

    print("\n[DONE] Use these outputs to decide how to adapt the prompt in agent_hybrid.py")


if __name__ == "__main__":
    main()
