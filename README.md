# Attune

Attune tunes a small language model to the way one person likes to be spoken to. You read four short answers, mark the one you like best and the one you like least, and after five rounds you have a model that answers in your style, plus a picture of what that style is.

**Status:** early. The training method, a benchmark with simulated users and a judge model, and a first version of the app exist. The app runs locally; it is not deployed yet.

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
| `app/backend/` | The app's server: password gate, and a private relay to the GPU service on Modal |
| `app/frontend/` | The app's pages (React, Vite, Tailwind) |

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

### The app, locally

```
scripts/app.sh mock    # a recorded session: no GPU, no cost; for working on the pages
scripts/app.sh         # real sessions on the Modal GPU (deploy modal_live.py first)
```

Open http://localhost:5173. The password is `attune` unless `ATTUNE_PASSWORD` is set. At the end of a real session the tuned adapter can be downloaded as a zip with a script to run it.

## Limits

- The simulated users score answers with hand-written word lists, and the fact check is keyword matching.
- Results depend on how each style is worded, and training can exaggerate a style late in a session.
- The answers are a style demonstration, not advice.
