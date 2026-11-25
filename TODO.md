# To do

Open work for Attune, grouped by area and roughly in priority order within each group.

## Next up

- [ ] Flatten line breaks in answers. About 2% come out one sentence per line (mostly the directive style); the trimming step used to remove them and has kept them since the two-paragraph experiment on 2026-11-25.
- [ ] Never show two identical answers in a round: redraw one, and if it repeats, give it a different style. Seen in 5 of about 180 benchmark rounds, from round 4 on, between styles differing on one axis (usually formality) where the answer is nearly the fact sheet verbatim.
- [ ] Run two or three sessions on the unchanged live page to see whether the same person gets the same profile twice.
- [ ] Have a few friends try the live page; note which axes they notice and whether any two feel the same.
- [x] Judge model, first version (`bench/judge.py`, Gemini 3.8 Flash, key in `.judge_key`): rates the four axes 1 to 5, checks each fact, lists additions.
- [ ] Hand-label about 30 answers and check the judge agrees, depth especially (15 of 22 against the requested style).
- [ ] Use the judge as the simulated user: the word lists cannot see formality (7 of 20 runs right, below chance) and are weak on depth and directiveness. Needs the key stored as a Modal secret.
- [x] GRPO against DPO on the setup of 2026-11-25 (grid `full`: 4 simulated users x 3 seeds, 5 rounds, judge as user, best-and-worst feedback for both). Axes on the wanted side, of 4: untouched 1.50, DPO 2.83 (lr 1e-5) and 3.17 (3e-5), GRPO 3.25 (1e-5) and 3.67 (3e-5). At a matched learning rate GRPO's answers suited the user better in 10 and 11 of 12 paired runs. GRPO at 1e-5 (the live setting) against DPO at its best rate is not clearly ahead. Rerun if the model, menu or update changes.
- [x] Live app: learning rate 3e-5 for the 5-question session (matches the benchmarked setting), 1e-5 for the 10-question one.
- [ ] Check the 10-question session at 3e-5 with the current settings; the choice to keep it gentle rests on earlier runs.

## Benchmark and method

- [x] Simulated users for the current four axes (word lists; see the judge item above for their limits).
- [x] First benchmark of the four axes, 20 runs (grid `gentle`): warmth right in 20 of 20, directiveness 15, depth 14, formality 7.
- [ ] Measure whether working one round ahead hurts learning; if so, try moving only the writing of answers ahead.
- [ ] Make the update gentler. First test (4 runs per version, one seed): clipping the style choice as one action (`--choice-clip 0.2`) with `--lr 1e-5` cut the top style's probability after round one from 0.84 to 0.26 and kept wording drift at the untouched model's level, with no loss in axes learned. Still to do: second seed, rerun with the judge as the user, then switch the live app.
- [ ] No version recovered when the simulated user reversed every preference after round five (five more rounds). Find out whether that is the update or the test.
- [ ] Measure what the weights add beyond the four style choices: answer the same questions with the tuned model and with the untouched model given the same chosen style, and compare (judge preference, and how different the wording is).
- [ ] Add baselines a reviewer would ask for: a bandit over the menu with no fine-tuning, and DPO learning the menu choice.
- [ ] Check the style menu on one or two non-Qwen base models of similar size, and confirm their licences.
- [ ] Speed up the update by training on all candidates in one batch on the L40S; find out why live updates take about 11 s when the timing run took 6.5 s.
- [ ] Test longer sessions (more than six rounds) once the update is gentler.
- [ ] Give feedback an absolute signal: a "none of these" option, or one answer per round from the untouched model as an anchor, so the best of a bad set is not rewarded.
- [ ] Let rounds shift gradually from style to wording, with no separate phase: draw each answer's style from the model's current preferences with repeats allowed, and shrink random exploration as confidence grows (never to zero). Needs the gentler update first, and a check that same-style answers differ enough in wording.
- [ ] Idea only, not planned: a continuous dial per axis by steering the model's internal activations instead of using written instructions.
- [ ] A person may like more than one combination (for example warm with formal, and neutral with casual). Decision (2026-11-25): keep it simple; a session finds one style the person likes, not all of them. Display only: once the update is gentler, list a runner-up combination in the results when it is close to the top one. No search for multiple peaks.
- [ ] Repeatability is not a pass criterion (decided 2026-11-25): five sessions by the user gave five different profiles, and that is acceptable for a quiz, as with personality tests. The claim that the method learns a known preference rests on the benchmark with simulated users. Frame results as "your style in this session".
- [x] Profile table on the test page: "no clear preference" under 75%, a lean up to 95%, stated plainly above; plus a 12-round option.
- [ ] Run a QLoRA comparison: the same sessions with the model loaded in 4-bit and at full precision, reporting memory used, seconds per round, and any change in results.
- [ ] Add an "axes right" summary to `bench/report.py` so analyses stop living in one-off scripts.

## Style menu

- [ ] Watch for overlap between casual and warm, and between collaborative and "explains why".
- [ ] Deal with warm and "explains why" answers running past three sentences before the cap.
- [ ] Remove the retired axes (brevity, language, energy) and old grids from `data/advice.json` and `bench/remote.py`.
- [ ] Write a patient-communication scenario with its own fact sheets, then others (education, legal, finance, consulting).

## Real app

- [x] Backend (`app/backend`): password check, signed session cookie, and a private relay to Modal. Runs locally; a Dockerfile for Railway exists but is not deployed or tested there.
- [x] Front end (`app/frontend`), first version: sign-in, welcome, questions, profile with four scales, before and after, round-by-round charts.
- [ ] Serve the page from Railway so it loads instantly while the GPU starts; add loading states for the cold start.
- [ ] "Try your model" on the results screen: the visitor types any text or question and sees the standard model's version beside their own model's, as with the prepared examples. This is the in-app form of converting text. Needs one endpoint that runs both models while the session's GPU is still up (it shuts down five minutes after last use, so the box must say when it has expired), a decision on prompt wording for rewording against answering, and a length limit on the input.
- [ ] Honest wait indicators: tell the visitor what they are waiting for and why, from the server's real state, not a timer. Stages to report: GPU starting, model loading, writing the first answers; training on the last pick; writing the next answers; checking them (redraws); building the results; packaging the model. Needs the Modal service to expose its current stage (a status endpoint the page polls, or streamed progress).
- [ ] Enforce the session time limit on the server.
- [ ] Results screen: present the outcome as a profile, show the user's style against its opposite, keep the per-round charts and tracked answers.
- [ ] Add a visible note that the answers are a style demonstration, not advice.
- [ ] Visitors' own questions (core, decided 2026-11-25; nothing pre-selected is forced on them): a start screen with editable suggested questions, all entered up front. While the GPU starts, Gemini screens each question and writes its two-point fact sheet.
- [ ] Live content check by the judge, as a filter (never a rewrite): optional, not required. Measured 2026-11-25 (8 runs each, gentler update, judge as user): with the keyword check off, shown answers missing a fact rose from about 7% to about 14%, learning was unchanged (3.62 against 3.50 axes right) and content did not collapse. Not tested: a user who prefers shorter answers, which is where dropping content would be rewarded. Gemini took 6 to 7 s per call that day.
- [ ] About 30% of answers "add something" by the judge's standard, before and after training alike. Decide how strict that standard should be when hand-labelling.
- [ ] Decide what counts as a question the app accepts (requests for advice; the style menu does not fit trivia or creative writing) and what it declines.
- [ ] Check that generated fact sheets are good enough: short, two distinct points, no risky specifics.
- [x] Adapter download: packaged on Modal after the last round (adapter, profile, run script), kept for a day.
- [ ] Try the downloaded package on a real machine: run `example.py` and check the output matches the session.
- [ ] Never deploy while a session is in progress; a deploy drops it.

## Advanced options (later)

- [ ] Let visitors continue from a saved adapter: an upload (safetensors only, validated) or a server-side resume code.
- [ ] Let visitors choose the number of rounds, within tested limits.
- [ ] Let visitors choose which axes to tune. With fewer styles than answers per round, sampling must allow repeats.
- [ ] A "convert" mode that skips the quiz: upload a saved adapter, paste text, and get it reworded straight away. Depends on the adapter download and upload items above, and on the file recording its base model, menu version and prompt setup.
- [ ] Later, if convert gets used: saved per-setting profiles and a content check that works on pasted text.

## Write-up and resume

- [ ] Benchmark page: GRPO against DPO, what failed and why (reward-model GRPO, DPO collapse), and how this departs from textbook GRPO.
- [ ] Fill the resume bullets from real runs; never name a baseline without a number against it.
- [ ] Settle the title: "Attune — A Real-Time RLHF Platform for Fine-Tuning an LLM to a User's Communication Style" ("Preferred" optional; "Platform" is fair once the app is live).
- [ ] Expand the README before making the repo public.

## Tooling

- [ ] Fix `bench.remote status`: it prints "waiting" for jobs that were never launched as well as for queued ones.
- [ ] Fix the `launch` message that reports not-yet-launched jobs as "already done".
- [ ] After a deploy, wait for old containers to stop before launching or testing; requests sent straight away land on stale code.
