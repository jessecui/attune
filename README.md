# Attune

Attune tunes a small language model to the way one person likes to be spoken to. You say who is advising you and about what (a doctor, a tutor, a manager), read four short answers to each of five questions, mark the best and the worst, and end up with a model that answers in your style, which you can download.

**Status:** the method is frozen and benchmarked, and the app runs locally against a GPU service on Modal. It is not deployed publicly yet.

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
- **Content:** every question has a two-point fact sheet, written by hand for the built-in questions and by Gemini for a visitor's own setting. An answer missing a point is redrawn before you see it.
- **Result:** at the end, a judge model (Gemini) rates the tuned model's answers to six unseen questions on each scale, beside the standard model's.

## Results

Benchmarked with simulated users played by the judge model, five feedback rounds per run, rated on six unseen questions.

| Comparison | Result |
|---|---|
| GRPO against the untouched model | 91% answer-level win rate (104 wins, 11 ties, 5 losses) |
| GRPO against DPO, 28 matched runs | 57% win rate (124 wins, 73 ties, 83 losses); scales right 88% against 80% |
| Answers adding something not on the fact sheet | 14% for GRPO, 28% for DPO |

Ties count as half a win. On the four standard simulated users the two methods tie; GRPO's lead comes from users who care about only two scales or pick at random part of the time. DPO follows a preference the menu has no scale for (shorter answers) further. Details and every run are in `TODO.md`.

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

Open http://localhost:5180. The access prompt asks for the creator's name, set as `CREATOR_NAME` in `app/backend/.env` (copy `.env.example`); case and spaces are ignored. At the end of a real session the tuned adapter can be downloaded as a zip with a script to run it.

## Limits

- The judge model rates style and content; it has not yet been checked against hand labels. The live fact check before an answer is shown is keyword matching.
- Only advisers. A test with speakers who are not advising (a student, a client) kept the styles but lost content, so the app asks who is advising you.
- Results depend on how each style is worded, and training can exaggerate a style late in a session.
- The answers are a style demonstration, not advice.
