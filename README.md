# MARCH-Orz

MARCH-Orz is a Python-based autonomous **vision browser agent** that uses:

- **Playwright** for browser control
- **Qwen3-VL (via Ollama)** for reasoning, perception, and verification
- **OpenCV** for coordinate refinement (“sniper” click targeting)
- **EasyOCR** for text-aware element recovery

The project is organized around a multi-role loop:

> **Observe → Reason → Perceive → Execute → Verify → Memory**

---

## Features

- **Autonomous browser interaction**
  - Navigates pages, clicks elements, types text, and submits actions.
- **Three-role policy architecture**
  - **Reasoner** (text-only): decides next action.
  - **Perceiver** (vision): locates element coordinates.
  - **Verifier** (vision): checks whether expected UI change happened.
- **Robust coordinate parsing**
  - Handles multiple coordinate output formats from model responses.
- **OpenCV “sniper” refinement**
  - Converts rough normalized coordinates into more precise pixel targets.
- **OCR-assisted fallback**
  - EasyOCR-powered local text search when direct visual targeting is unreliable.
- **Structured memory**
  - Episodic records + periodic compression for long-horizon task tracking.
- **Goal decomposition**
  - Planner generates step-by-step subgoals from a high-level objective.
- **Secure data injection via vault**
  - Supports `$key` placeholders resolved from `vault.json`.

---

## Repository Structure

- `agent_loop.py` — Main structured autonomous loop (recommended entry point)
- `policy.py` — Reasoner / Perceiver / Verifier LLM policy layer
- `planner.py` — Goal-to-subtask decomposition
- `memory.py` — Structured memory, episode tracking, compression
- `browser_engine.py` — Playwright wrapper (`VisionBrowser`)
- `ocr_sniper.py` — EasyOCR-based local text targeting helper
- `perception.py` — State extraction and UI element detection helpers
- `agent.py` — Simpler demo-style loop using Qwen + OpenCV
- `vault.json` — Secure value map for runtime text injection (`$key`)
- `agent_loop_V1.py` — Older iteration of the loop
- `dom_test.py` — Experimental/testing utility
- `MARCH Orz/` — Duplicate/alternate code snapshot folder

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- Qwen vision model available in Ollama (used in this repo):
  - `qwen3-vl:8b-instruct`
  - (and in some scripts) `qwen3-vl:4b-instruct`
- Playwright + browser binaries
- OpenCV
- EasyOCR (GPU enabled in code by default)
- Pillow
- NumPy

---

## Installation

```bash
git clone https://github.com/kamikaze-san/MARCH-Orz.git
cd MARCH-Orz
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, install manually:

```bash
pip install playwright pillow numpy opencv-python easyocr ollama
playwright install chromium
```

---

## Ollama Setup

Pull required models:

```bash
ollama pull qwen3-vl:8b-instruct
ollama pull qwen3-vl:4b-instruct
```

Start Ollama service (if not already running), then run the agent.

---

## Quick Start

Run the main autonomous loop:

```bash
python agent_loop.py
```

By default, `agent_loop.py` contains a built-in test goal and start URL in `__main__`.
Edit those values to run your own workflow:

- start URL
- natural language goal
- max step count

---

## Secure Data Vault

The agent supports typing secure values through placeholders.

In actions, if text is `$email`, the agent looks up `email` in `vault.json` and injects the real value at runtime.

Example `vault.json`:

```json
{
  "email": "user@example.com",
  "password": "super-secret-password",
  "pincode": "123456"
}
```

Use with care. Never commit real credentials to public repos.

---

## How It Works (High Level)

1. **Observe**
   - Capture current screenshot in memory.
2. **Reason**
   - Text-only model decides *what* to do next and expected outcome.
3. **Perceive**
   - Vision model finds *where* target element is.
4. **Execute**
   - Convert normalized coords to pixels, refine via OpenCV, then click/type.
5. **Verify**
   - Compare before/after screenshots to detect success/failure.
6. **Memory**
   - Record episode, compress older history, progress through plan steps.

