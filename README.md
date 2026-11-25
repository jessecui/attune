# Attune

Attune tunes a small language model to the way one person likes to be spoken to. You read a few short answers, mark the one you like best and the one you like least, and after six rounds you have a model that answers in your style, plus a picture of what that style is.

**Status:** early. The training method, a benchmark with simulated users, and a bare test page for live sessions exist. The real app does not yet.

## How a session works

1. You get a question, such as "I keep forgetting to take my daily medication. What can I do?"
2. The model writes four answers of up to three sentences. Each is written in a different combination of styles.
3. You mark your favourite and your least favourite.
4. The model is updated on that choice, and the next round begins.
5. After six rounds you see how the model's preferences moved, the same question answered after every round, and new questions answered before and after.

The styles come from a fixed menu of four axes:

| Axis | One end | Other end |
|---|---|---|
| Warmth | Opens by recognising how you may feel | Goes straight to the advice |
| Directiveness | Tells you what to do | Describes what works and leaves the decision to you |
| Formality | Relaxed and conversational | Formal |
| Depth | Explains why | Just the advice |

Every question comes with a short fact sheet, and every answer must contain those facts. The aim is to change how something is said without changing what is said.

## How it learns

- **Model.** [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with a LoRA adapter. Only the adapter is trained.
- **Update.** GRPO: the four answers are a group sampled from the current model, your best and worst choices become rewards of +1 and −1, and the model takes a few clipped policy-gradient steps.
- **Two-step policy.** The model first chooses a style from the menu, then writes its answer to a prompt that carries that style as a bulleted list in the user turn. Both steps are trained by the same feedback.
- **Neutral start.** The style choice is measured relative to the untouched model, so every style starts equally likely.
- **Content checks.** An answer missing a fact is redrawn before you see it, and answers are cut to three sentences unless that would drop a fact.
- **Working ahead.** In a live session the server applies your previous choice and writes the next answers while you read, so there is almost no wait between rounds.

This departs from textbook GRPO in ways that matter: the reward comes from a person, the group is four, about 30% of candidate styles are drawn at random for variety, the style choice is limited to a menu, and working ahead means each update is applied one step late.

## What is in the repo

| Path | What it is |
|---|---|
| `bench/run.py` | One simulated session: candidates, feedback, update, evaluation |
| `bench/methods.py` | The update rules: GRPO with human reward, plus DPO and reward-model GRPO for comparison |
| `bench/policy.py` | The model wrapper: generation, log-probabilities, adapter snapshots |
| `bench/personas.py` | Simulated users and the text features they score |
| `bench/check_menu.py` | Checks whether a model follows each option on a style menu |
| `bench/remote.py`, `modal_app.py` | Launch benchmark jobs on Modal and collect results |
| `bench/report.py` | Aggregate results across seeds |
| `modal_live.py`, `live_page.html` | The live session service and a plain test page |
| `data/` | Scenarios: questions, fact sheets and style menus |
| `scripts/feed.sh` | Launch a set of jobs a few at a time |

## Running it

Set up a Python environment:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

A single simulated session runs locally on a GPU or an Apple-silicon Mac. A smaller model is quicker for a first try:

```
.venv/bin/python -m bench.run --model Qwen/Qwen2.5-1.5B-Instruct \
  --method grpo_style --two-step --neutral-start --feedback bestworst \
  --scenario data/advice.json --axes warmth,directiveness,formality,depth \
  --persona gentle_guide --rounds 6 --max-new-tokens 96 --max-sentences 3 \
  --min-facts 1.0 --strict-facts --eval-split test
```

Everything else runs on [Modal](https://modal.com). After `modal setup`:

```
.venv/bin/modal deploy modal_app.py                # the benchmark app
.venv/bin/python -m bench.remote launch neutral    # start a set of runs; returns at once
.venv/bin/python -m bench.remote status neutral    # progress, and download finished results
.venv/bin/python -m bench.report results/modal/neutral
```

The sets of runs are defined in `GRIDS` in `bench/remote.py`.

For a live session:

```
.venv/bin/modal deploy modal_live.py
```

The deploy prints a URL and writes an access key to `.live_key`. Open the URL with `?key=<the key>` appended. The container shuts down five minutes after the last request.

## What the benchmark has shown so far

These come from small samples, usually six simulated sessions per setting, and should be read as indications.

- **Where the style instructions go matters more than model size.** Written as the opening line of the model's own reply, several instructions at once were mostly ignored, even by the 7B model. As a bulleted list in the user turn, they were followed.
- **Short answers that keep their facts needed the 7B model.** A 1.5B model followed tone but could not control length or fit the facts into a short styled answer.
- **Marking best and worst worked as well as ranking all four.**
- **A neutral start learned at least as well as starting from the model's defaults,** and faster.
- **DPO learned preferences well but sometimes collapsed,** dropping the content entirely when the simulated user rewarded brevity. A rollback guardrail stopped the collapses at the cost of most of the learning.
- **An L40S was both the fastest and the cheapest per session** of the GPUs tried.

## Limits

- **The simulated users are crude.** They score answers with hand-written word lists, and they predate the current four axes. A judge model should replace them.
- **The fact check is keyword matching.** It catches missing facts, not invented ones, and the model does sometimes add details.
- **Training can exaggerate a style.** Late in a session, wording can drift past what the instruction asked for.
- **Results depend on wording.** People choose between sentences, not labels, so a poorly worded option can lose for the wrong reason.
- **The adapter depends on the scaffold.** It reproduces a style only with the same menu, prompt format and neutral-start calculation.
- **The answers are a style demonstration, not advice.**

## Next

- A judge model for scoring style and checking content.
- A GRPO and DPO comparison on the current setup, across seeds.
- The real app: a web front end and a thin backend, with Modal as the GPU service.
