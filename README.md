# Attune

Attune fine-tunes a 7B language model live on one person's feedback, so it learns how they like an advisor to speak to them. You say who is speaking to you and about what ("my doctor explaining my test results"), mark the best and worst of four answers to each of five questions, and get back a model trained on your picks, which you can download and run yourself.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshot-round-dark.png">
  <img alt="A training round: four answers to &quot;What is the best way to choose my race shoes?&quot; from a running coach, each in a different style, with one marked best and one marked worst" src="docs/screenshot-round-light.png">
</picture>

## Features

- 🎯 **Online RLHF**: GRPO on a LoRA adapter, with your best and worst picks as the reward; one update per question, about five minutes for a session
- 🧭 **Four style scales**: empathy, directiveness, formality and depth, each answer drawn from a menu of style combinations
- 🩺 **Any advisor, any topic**: Gemini turns your description into five questions and a two-point fact sheet for each, so every answer gives the same advice and only the style differs
- ✅ **Content check**: an answer missing a fact-sheet point is redrawn before you see it; near-identical answers are replaced
- 📏 **Measured result**: a judge model rates your tuned model on five questions it was not trained on, beside the standard model
- 💬 **Try it**: ask your own question and compare the standard model's answer with yours
- 📦 **Download**: the adapter as a zip, with your results and a script to run it
- 🌓 **Light and dark themes**

## Results

Benchmarked with simulated users played by the judge model (each given a fixed set of preferences), five feedback rounds per run, rated on unseen questions:

| Comparison | Result |
|---|---|
| GRPO against the untouched model | **96%** answer-level win rate (113 wins, 4 ties, 3 losses) |
| GRPO against DPO, 28 matched runs | **59%** win rate (122 wins, 88 ties, 70 losses) |
| Scales on the wanted side | 87% for GRPO, 72% for DPO |
| Run-to-run variance in scales learned | **73% lower** for GRPO |
| Answers adding advice not on the fact sheet | 26% for GRPO, 22% for DPO |

Ties count as half a win. GRPO leads on the four standard simulated users (59% head to head) and most clearly on a user who cares about only two scales (94% of those scales right, against 75%). It does not reduce additions: its answers add something not on the fact sheet slightly more often than DPO's. An earlier run had shown the opposite, because answers written in a style were cut at three sentences while DPO's could use four; with the same limit for both, that gap disappeared. In an earlier run with a user who picks the shortest answer, DPO, which trains on the text directly, shortened its answers further than GRPO.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshot-results-dark.png">
  <img alt="The results page: what the model learned on each of the four scales, rated from its writing, with the standard model's rating shown as a hollow marker" src="docs/screenshot-results-light.png">
</picture>

## Architecture

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "15px", "lineColor": "#94a3b8", "textColor": "#64748b", "clusterBkg": "transparent", "clusterBorder": "#94a3b8", "titleColor": "#64748b"}, "flowchart": {"curve": "basis", "nodeSpacing": 45, "rankSpacing": 40, "padding": 20, "wrappingWidth": 400}}}%%
flowchart TB
    UI(["Browser<br/>advisor + topic"])

    D["Draft the setting<br/>5 questions + fact sheets · Gemini<br/>Modal CPU function"]

    subgraph APP["App server (FastAPI)#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;"]
        A["Name gate + private relay<br/>signed cookie"]
    end

    subgraph LIVE["Live session (Modal, L40S GPU)#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;#nbsp;"]
        C["Choose 4 styles<br/>from the style menu"]
        W["Write 4 answers<br/>Qwen2.5-7B + LoRA"]
        F["Check facts<br/>redraw if a point is missing"]
        PICK(["Browser<br/>mark best + worst"])
        U["GRPO update<br/>best and worst as reward<br/>repeat for each of 5 questions"]
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

**How the update works.** Training happens online, with the adapter updated after every question during the session. The model makes two choices for each answer: first a style line from the menu, then the answer written in that style. GRPO compares the four answers to one question, so the best pick's choices are reinforced and the worst's discouraged. The style choice is clipped as a single action (ratio 0.2), the wording is trained at 0.3 of that strength so the model doesn't settle on one stock phrase, and 30% of styles are drawn at random to keep exploring. The four styles in a round differ on at least two scales.

**Benchmark.** The same session code runs on Modal with a simulated user in place of the browser: the judge model, given a persona, picks best and worst. GRPO and DPO (trained on the same best-and-worst pairs) run on matched personas and seeds, and the judge rates each run's answers to unseen questions.

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
.venv/bin/python -m bench.remote launch final      # start a set of runs; returns at once
.venv/bin/python -m bench.remote status final      # progress; downloads finished results
.venv/bin/python -m bench.judge compare final      # judge the runs' answers and summarise
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
- Results depend on how each style line is worded. Empathetic answers often share an opening ("It can feel…"): the untouched model does the same when given that style, so it comes from the instruction and not from training.
- The answers demonstrate a style. They can be wrong and are not medical, legal or financial advice.
