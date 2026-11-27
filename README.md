# Attune

Attune fine-tunes a 7B language model live on one person's feedback, so it learns how they like an advisor to speak to them. You say who is speaking to you and about what ("my doctor explaining my test results"), mark the best and worst of four answers to each of eight questions, and get back a model trained on your picks, which you can download and run yourself. The finished model needs no style prompt: ask it a plain question and it answers in the style it learned.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshot-round-dark.png">
  <img alt="A training round: four answers to &quot;What should I do when I get stuck on a difficult concept?&quot; from a tutor, each in a different style, with one marked best and one marked worst" src="docs/screenshot-round-light.png">
</picture>

## Features

- 🎯 **Online RLHF**: GRPO on a LoRA adapter, with your best and worst picks as the reward; one update per question, about six minutes for a session
- 🪶 **No prompt scaffolding**: style instructions only make the four answers in a round differ. The training changes how the model answers a plain question, and the finished model is never given a style
- 🧭 **Four style scales**: empathy, directiveness, formality and depth, each answer drawn from a menu of style combinations
- 🩺 **Any advisor, any topic**: Gemini turns your description into eight questions and a two-point fact sheet for each, so every answer gives the same advice and only the style differs
- ✅ **Content check**: an answer missing a fact-sheet point is redrawn before you see it; near-identical answers are replaced
- 📏 **Measured result**: a judge model rates your tuned model on five questions it was not trained on, beside the standard model
- 💬 **Try it**: ask your own question and compare the standard model's answer with yours
- 📦 **Download**: the adapter as a zip, with your results and a script to run it
- 🌓 **Light and dark themes**

## Results

Benchmarked with simulated users played by the judge model (each given a fixed set of preferences), eight feedback rounds per run. Every answer scored is to a question the model was not trained on, written from a plain prompt with no style in it. The settings were chosen on one set of runs; these numbers come from a second set on seeds that played no part in the choice (12 runs per method):

| Comparison | Result |
|---|---|
| GRPO against the untouched model | **93%** answer-level win rate (107 wins, 10 ties, 3 losses) |
| GRPO against DPO, same users and seeds | 45% (31 wins, 45 ties, 44 losses); **50%** over both sets of runs (75, 88, 77) |
| Scales on the wanted side | 83% for GRPO, 88% for DPO |
| Fact-sheet points covered | 100% for GRPO, 98% for DPO |
| Answers adding advice not on the fact sheet | 36% for GRPO, 39% for DPO |

Ties count as half a win. GRPO and DPO are level: each led by a few points on one set of runs, and over both sets the head-to-head is 50%. Both were tuned the same way, with the same learning rates, users and seeds, and both reuse earlier rounds' picks in later updates.

What the tuning runs showed:

- **Session length.** The judge's fit score for GRPO rose from 3.83 after five rounds to 4.84 after eight and was no higher after ten (4.69), so sessions are eight questions.
- **Experience replay.** Spending half of each update on earlier rounds' picks makes GRPO learn faster. After five rounds its win rate over the untouched model was 93% with replay and 85% without (67% head to head). By eight rounds the version without it had nearly caught up (90% against 93%).
- **Learning rate.** At 2e-5 no content was lost. At 1e-4 a third of answers dropped a fact, for both methods.
- **The model's own answers.** Replacing one or two of the four styled answers with the model's own unprompted answer learned worse at every session length: two styled answers do not give enough variety to choose between.
- **Longer sessions and content.** By ten rounds 8% of DPO's answers missed a fact, against 1% of GRPO's.

An earlier version of Attune learned which of sixteen style instructions to use and wrote the chosen one into the model's prompt. It scored well (a 96% win rate over the untouched model), but with that instruction removed the tuned model wrote like the untouched one (fit −0.16 against 0.00), so it learned a prompt, not a way of writing. The method above replaced it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshot-results-dark.png">
  <img alt="The results page: what the model learned on each of the four scales, rated from its writing, with the standard model's rating shown as a hollow marker" src="docs/screenshot-results-light.png">
</picture>

## Architecture

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "15px", "lineColor": "#94a3b8", "textColor": "#64748b", "clusterBkg": "transparent", "clusterBorder": "#94a3b8", "titleColor": "#64748b"}, "flowchart": {"curve": "basis", "nodeSpacing": 45, "rankSpacing": 40, "padding": 20, "wrappingWidth": 400}}}%%
flowchart TB
    UI(["Browser<br/>advisor + topic"])

    D["Draft the setting<br/>8 questions + fact sheets · Gemini<br/>Modal CPU function"]

    subgraph APP["App server (FastAPI)#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;"]
        A["Name gate + private relay<br/>signed cookie"]
    end

    subgraph LIVE["Live session (Modal, L40S GPU)#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;"]
        C["Draw 4 styles<br/>for variety only"]
        W["Write 4 answers<br/>Qwen2.5-7B + LoRA"]
        F["Check facts<br/>redraw if a point is missing"]
        PICK(["Browser<br/>mark best + worst"])
        U["GRPO update on the plain prompt<br/>best and worst as reward<br/>repeat for each of 8 questions"]
        M["Measure<br/>5 unseen questions · rated by Gemini"]
        PKG["Package the adapter<br/>download served by a Modal CPU function"]
        C --> W --> F --> PICK --> U
        U -.-> C
        U --> M
        U --> PKG
    end

    OUT(["Browser<br/>results · try it · download"])

    UI --> A --> D --> C
    M --> OUT
    PKG --> OUT

    classDef step fill:#f0fdfa,stroke:#94a3b8,stroke-width:1px,color:#0f172a
    classDef model fill:#ccfbf1,stroke:#0f766e,stroke-width:2px,color:#0f172a
    classDef client fill:#0f766e,stroke:#0f766e,color:#ffffff

    class A,D,C,F,M,PKG step
    class W,U model
    class UI,PICK,OUT client
    linkStyle default stroke:#94a3b8,stroke-width:1.5px
```

**How the update works.** Training happens online, with the adapter updated after every question during the session. Each round's four answers are written under four different style instructions drawn from the menu. The instructions are there only to make the answers differ: each answer is then scored as a reply to the plain question, with no style named. GRPO compares the four, so the best pick becomes more likely as a plain answer and the worst less likely, with every step clipped (ratio 0.2). Half of each update's steps go back over earlier rounds' picks, so each pick is used more than once. Because the answers are written under style instructions and not sampled from the plain prompt itself, this is GRPO's objective used off-policy. The tuned model is never given a style: on the results page, in the try box and in the download, its prompt is the question alone.

**Benchmark.** The same session code runs on Modal with a simulated user in place of the browser: the judge model, given a persona, picks best and worst. GRPO and DPO are trained on the same best-and-worst picks, on matched personas and seeds, with the same learning rates tried for each. A method's settings are chosen on held-out questions and one set of seeds, and the comparison is reported on test questions and fresh seeds.

## Tech Stack

- **Frontend**: Vite 7, React 19, TypeScript, Tailwind CSS v4
- **Backend**: FastAPI, Python
- **Model**: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with a LoRA adapter (r = 16), PyTorch, Transformers, PEFT
- **Training**: GRPO with human best/worst feedback; DPO as the baseline
- **GPU**: [Modal](https://modal.com), one L40S for live sessions, more for benchmarks
- **Drafting and judging**: Gemini 3.8 Flash, falling back to 3.6 Flash and then 3.5 Flash

---

## Quick Start

### Prerequisites

- Python 3.10+, Node.js 20.19+ or 22.12+
- A [Modal](https://modal.com) account (for real sessions and benchmarks)
- A Google AI Studio API key for Gemini, saved as `.judge_key` and as the Modal secret `attune-judge` (`GEMINI_API_KEY`)

### 1. Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r app/backend/requirements.txt
cd app/frontend && npm install
```

### 2. Configure

Copy `app/backend/.env.example` to `app/backend/.env` and set `CREATOR_NAME`, the name the access prompt asks for (case and spaces are ignored).

The sign-in gate is in `app/backend/auth.py`: the right name sets a signed session cookie that lasts 24 hours, and ten wrong names from one address within 15 minutes lock that address out until the window passes. Its tests run with `cd app/backend && ../../.venv/bin/python -m unittest test_auth`.

### 3. Run the app

```bash
scripts/app.sh mock    # a recorded session: no GPU, no cost
scripts/app.sh         # real sessions on Modal (deploy the live service first, below)
```

Open http://localhost:5180. The backend runs on port 8787.

### 4. The GPU service and benchmarks

```bash
.venv/bin/modal deploy modal_live.py               # live session service; writes its access key to .live_key
.venv/bin/modal deploy modal_app.py                # benchmark app
.venv/bin/python -m bench.remote launch confirm    # start a set of runs; returns at once
.venv/bin/python -m bench.remote status confirm    # progress; downloads finished results
.venv/bin/python -m bench.score confirm 8          # judge the runs' answers after round 8 and summarise
```

For real sessions, `scripts/app.sh` also needs the two addresses that `modal deploy modal_live.py` prints and a Modal proxy token in `app/backend/.env` (see `.env.example`). The services refuse any request without the token before a container starts:

```bash
.venv/bin/modal workspace proxy-tokens create --name attune
```

---

## Project Structure

| Path | What it is |
|---|---|
| `bench/` | Session simulation, GRPO and DPO, simulated users, the judge, setting drafts |
| `data/` | Built-in questions, fact sheets and the style menu (`advice.json`); `experiments/` holds the copies used in wording tests |
| `modal_live.py` | The live session service: generation, updates, results, try box, adapter packaging |
| `modal_app.py`, `bench/remote.py` | Benchmark jobs on Modal |
| `app/backend/` | FastAPI server: name gate, private relay to Modal, mock mode |
| `app/frontend/` | The pages (React, Vite, Tailwind) |
| `scripts/` | `app.sh` to run the app locally, `feed.sh` to queue benchmark runs |
| `docs/` | Screenshots |

## Limits

- Advice only. In a test with speakers who aren't advising (a student, a client), the styles came through but content was covered less reliably: both points appeared in 62% of the student's answers and 84% of the client's, against 90% for an advisor. So the setting step keeps every session to advice: it recasts the role as someone advising you, and declines settings it can't recast.
- The judge rates style and content and has been spot-checked by eye, not against a full set of hand labels. It is loosest on depth, sometimes counting a short "to…" phrase as a reason. The live fact check before an answer is shown is keyword matching.
- Results depend on how each style line is worded. The model learns from answers written under those lines, so it can inherit their habits, such as empathetic answers that open the same way ("It can feel…").
- Five picks teach less than eight: after five rounds GRPO's fit was 4.11, against 4.72 after eight, on the same runs.
- The answers demonstrate a style. They can be wrong and are not medical, legal or financial advice.
