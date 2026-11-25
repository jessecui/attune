# Attune

Attune tunes a small language model to the way one person likes to be spoken to. You read four short answers, mark the one you like best and the one you like least, and after six rounds you have a model that answers in your style, plus a picture of what that style is.

**Status:** early. The training method, a benchmark with simulated users, and a bare test page for live sessions exist. The real app does not yet.

## How it works

Each answer is written in a combination of styles from a fixed menu:

| Axis | One end | Other end |
|---|---|---|
| Warmth | Opens by recognising how you may feel | Goes straight to the advice |
| Directiveness | Tells you what to do | Describes what works and leaves the decision to you |
| Formality | Relaxed and conversational | Formal |
| Depth | Explains why | Just the advice |

- **Model:** [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with a LoRA adapter.
- **Update:** GRPO, with your best and worst choices as the reward. The model first chooses a style, then writes the answer in it; both steps are trained.
- **Content:** every question has a short fact sheet, and an answer missing a fact is redrawn before you see it.

## Layout

| Path | What it is |
|---|---|
| `bench/` | Simulated sessions, the update rules (GRPO, plus DPO for comparison), simulated users, and a check of whether a model follows each menu option |
| `data/` | Questions, fact sheets and style menus |
| `modal_app.py`, `bench/remote.py` | Run benchmark jobs on Modal and collect results |
| `modal_live.py`, `live_page.html` | The live session service and a plain test page |

## Running it

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Benchmark jobs and the live session run on [Modal](https://modal.com). After `modal setup`:

```
.venv/bin/modal deploy modal_app.py                # benchmark app
.venv/bin/python -m bench.remote launch neutral    # start a set of runs; returns at once
.venv/bin/python -m bench.remote status neutral    # progress, and download finished results

.venv/bin/modal deploy modal_live.py               # live session
```

The live deploy prints a URL and writes an access key to `.live_key`. Open the URL with `?key=<the key>` appended.

## Limits

- The simulated users score answers with hand-written word lists, and the fact check is keyword matching.
- Results depend on how each style is worded, and training can exaggerate a style late in a session.
- The answers are a style demonstration, not advice.
