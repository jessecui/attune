# To do

Open work for Attune, grouped by area and roughly in priority order within each group.

## Next up

- [x] Flatten line breaks in answers (done: the trimming step joins every answer into one paragraph). About 2% come out one sentence per line (mostly the directive style); the trimming step used to remove them and has kept them since the two-paragraph experiment on 2026-11-25.
- [x] Warm wording tested (2026-11-25): the current "open with a short remark" reads as warm in 92% of answers (4.3 against 1.1 for neutral) with a stock sympathy opening in 71%. "Write warmly, as someone who cares" lost the warmth (12%, 1.7). "Acknowledge how the person may be feeling, somewhere in the answer" was in between (54%, 3.0) and twice came out as an instruction to the reader. Kept the current wording; the repeated "I know…" in live sessions is the trained model settling on one phrase, addressed by the gentler-wording test.
- [x] The script in the downloaded package failed on current transformers (chat template returns a mapping, not a tensor). Fixed and verified on a GPU with `modal run modal_app.py::package`. Packages made before the fix have the broken script.
- [x] In-role sessions verified: the model speaks as the chosen role ("Inform me immediately when project timelines shift").
- [x] Wording collapse at 3e-5 (one stock opening, identical answers across styles in a live session) tested 2026-11-25, 12 runs each. Axes of 4 / fit / wording drift (standard model 0.30) / top style after round one: 1e-5: 3.25 / 4.78 / 0.31 / 0.23. 2e-5: 3.58 / 5.61 / 0.36 / 0.44. 3e-5: 3.67 / 5.94 / 0.62 / 0.61. Wording trained at 0.3 strength with 3e-5: 3.58 / 5.33 / 0.46 / 0.62. The in-between rate (the user's suggestion) kept the learning and brought drift back down; live five-question sessions now use 2e-5. Drawing the four so every scale shows both options made no measurable difference to learning (3.42 against 3.67 alone; 3.58 with gentler wording) and is not adopted.
- [x] Live setting from late 2026-11-25: learning rate 2e-5 with the wording trained at 0.3 strength, the four styles drawn at least two scales apart, near-copies replaced, and the results answers sampled (temperature 0.7) instead of greedy. Benchmark of 2e-5 with 0.3 wording (12 runs): 3.50 of 4 axes, fit 5.30, drift 0.33, same opening in 32% of final answers (2e-5 alone: 3.58, 5.61, 0.36, 40%). A scripted live session had no near-copy in five rounds, and the user's next session opened its three answers three different ways. The "two scales apart" rule itself has not been benchmarked.
- [x] Final benchmark of the frozen method (grid `final`, 2026-11-25 late; 12 runs per arm): GRPO as shipped (2e-5, wording at 0.3, styles two scales apart, near-copies replaced) 3.25 of 4 axes and fit 4.33; DPO at 2e-5 3.25 and 4.33; DPO at 3e-5 2.92 and 4.08; untouched model 2.00. Answer-level win rate of GRPO over the untouched model 91%, over DPO 53 to 55% (a tie), and no difference in run-to-run variance. So the earlier resume figures (75% win rate over DPO, 65% lower variance) hold for the 3e-5 setup they were measured on, not for what is shipped.
- [x] Less tidy simulated users (grid `users`, 2026-11-25 late; 8 runs per cell, GRPO as shipped against DPO at 2e-5). User who cares about two scales: GRPO had both right in 8 of 8 runs (100%), DPO 81%, standard model 50%; answers adding something 4% against 22%. User who picks at random a quarter of the time: 84% against 78% of scales right, win rate over the standard model 91% against 89%. User who picks the shortest answer: DPO shortened answers from 40 words to 13, GRPO to 23, and both kept the facts (99% and 97%), so neither collapsed. Reading: GRPO is at least as good on preferences the menu covers and stays closer to the fact sheet; DPO, which trains on the text directly, follows a preference the menu has no scale for (length) further. Eight runs per cell, so differences of a few points are not established.
- [x] Pooled result for the resume (2026-11-25): the 28 runs where GRPO as shipped and DPO at 2e-5 met the same simulated user and seed (`final` plus `users`). Head-to-head answer-level win rate 57% (124 wins, 73 ties, 83 losses; ties count half); scales on the wanted side 88% against 80%; answers adding something not on the fact sheet 14% against 28% (51% fewer); run-to-run variance 31% lower. Win rate over the untouched model 91% (`final`, 104 wins, 11 ties, 5 losses). On the standard users alone the two methods tie, so the 57% leans on the less tidy users. Rerun if the method changes; page wording does not matter.
- [x] Scope: speakers who are not advisers (branch `beyond-advice`, tag `before-beyond-advice` on main, grid `menu_check17`, 2026-11-25). A generalised menu and setting brief for a manager, a student and a client. Styles came through for all three (judge separation of 2 to 3 points per scale), but content held up only for the adviser: both points covered 90% manager, 84% client, 62% student; answers adding something 12%, 22%, 48%. The student kept slipping into the adviser's seat ("You may submit your paper on Monday"). Decision: keep the scope to advisers; the welcome page now asks "Who is advising you? What about?". The branch is kept, not merged.
- [x] Decide what to claim and what to ship (settled by the pooled result above; the method is frozen as shipped). Evidence so far: the same DPO setting scored 3.17 in one batch and 2.92 in the next, so a gap of 0.3 axes between 12-run arms is noise; forcing the four styles apart lowered GRPO's fit both times it was tried (5.94 to 5.11 as "spread", 5.30 to 4.33 as "two scales apart"), each within noise alone. Options: quote the tie honestly; ship without the two-scales-apart rule (keeping near-copy replacement) and re-measure; or run many more seeds to see whether any gap is real.
- [ ] The tuned model can still settle on one opening ("I know it can…" in all three results answers of a live manager session, late 2026-11-25, after a session where the three differed). The user called it not the biggest deal. If it is worth another pass: lower the wording strength below 0.3, or raise the randomness used for the results answers, and benchmark before switching.
- [ ] Answers in a round are sometimes too alike (user, 2026-11-25, after several live passes), and the user wanted more variety in how advice is put, such as "Try…". Two candidate changes, both needing a benchmark run: (1) draw the four styles so that every scale shows both of its options in each round, which makes the four differ more and gives a pick on every scale; (2) a third, middle option on directiveness for suggestions ("Try…", "You could…"), between commands and plain statements.
- [x] Never show two identical answers in a round: redraw one, and if it repeats, give it a different style. Seen in 5 of about 180 benchmark rounds, from round 4 on, between styles differing on one axis (usually formality) where the answer is nearly the fact sheet verbatim.
- [ ] Run two or three sessions on the unchanged live page to see whether the same person gets the same profile twice.
- [ ] Have a few friends try the live page; note which axes they notice and whether any two feel the same.
- [x] Judge model, first version (`bench/judge.py`, Gemini 3.8 Flash, key in `.judge_key`): rates the four axes 1 to 5, checks each fact, lists additions.
- [x] Skipped (2026-11-26): a full hand-label check. Spot-checked eight judged answers instead: warmth, directiveness and formality at the right end every time; depth generous with short "to…" phrases; additions uneven (one strict call, one missed). Enough for a side project; needed only to claim the judge is validated against hand labels. and check the judge agrees, depth especially (15 of 22 against the requested style).
- [x] Use the judge as the simulated user (`--judge-user`, secret `attune-judge`): the word lists cannot see formality (7 of 20 runs right, below chance) and are weak on depth and directiveness. Needs the key stored as a Modal secret.
- [x] GRPO against DPO on the setup of 2026-11-25 (grid `full`: 4 simulated users x 3 seeds, 5 rounds, judge as user, best-and-worst feedback for both). Axes on the wanted side, of 4: untouched 1.50, DPO 2.83 (lr 1e-5) and 3.17 (3e-5), GRPO 3.25 (1e-5) and 3.67 (3e-5). At a matched learning rate GRPO's answers suited the user better in 10 and 11 of 12 paired runs. GRPO at 1e-5 (the live setting) against DPO at its best rate is not clearly ahead. Rerun if the model, menu or update changes.
- [x] (Superseded: now 2e-5.) Live app: learning rate 3e-5 for the 5-question session (matches the benchmarked setting), 1e-5 for the 10-question one.
- [x] (Superseded: the app offers only 5 questions.) Check the 10-question session at 3e-5 with the current settings; the choice to keep it gentle rests on earlier runs.

## Benchmark and method

- [x] Simulated users for the current four axes (word lists; see the judge item above for their limits).
- [x] First benchmark of the four axes, 20 runs (grid `gentle`): warmth right in 20 of 20, directiveness 15, depth 14, formality 7.
- [ ] Measure whether working one round ahead hurts learning; if so, try moving only the writing of answers ahead.
- [x] Make the update gentler (shipped: choice clip 0.2, 2e-5, wording at 0.3). First test (4 runs per version, one seed): clipping the style choice as one action (`--choice-clip 0.2`) with `--lr 1e-5` cut the top style's probability after round one from 0.84 to 0.26 and kept wording drift at the untouched model's level, with no loss in axes learned. Still to do: second seed, rerun with the judge as the user, then switch the live app.
- [ ] No version recovered when the simulated user reversed every preference after round five (five more rounds). Find out whether that is the update or the test.
- [ ] Measure what the weights add beyond the four style choices: answer the same questions with the tuned model and with the untouched model given the same chosen style, and compare (judge preference, and how different the wording is).
- [ ] Add baselines a reviewer would ask for: a bandit over the menu with no fine-tuning, and DPO learning the menu choice.
- [ ] Check the style menu on one or two non-Qwen base models of similar size, and confirm their licences.
- [ ] Speed up the update by training on all candidates in one batch on the L40S; find out why live updates take about 11 s when the timing run took 6.5 s.
- [ ] Test longer sessions (more than six rounds) once the update is gentler.
- [ ] Give feedback an absolute signal: a "none of these" option, or one answer per round from the untouched model as an anchor, so the best of a bad set is not rewarded.
- [ ] Not built, and not planned while sessions are five questions (reviewed 2026-11-25 late): let rounds shift gradually from style to wording by drawing each answer's style from the model's preferences with repeats allowed and shrinking exploration as confidence grows. It was meant for sessions of 20 or 30 rounds. The shipped app goes the other way within a round (every pair of answers at least two scales apart), because same-style answers came out as near-copies. Revisit only if longer sessions or continuing a saved model come back into scope.
- [ ] Idea only, not planned: a continuous dial per axis by steering the model's internal activations instead of using written instructions.
- [ ] A person may like more than one combination (for example warm with formal, and neutral with casual). Decision (2026-11-25): keep it simple; a session finds one style the person likes, not all of them. Display only: once the update is gentler, list a runner-up combination in the results when it is close to the top one. No search for multiple peaks.
- [ ] Repeatability is not a pass criterion (decided 2026-11-25): five sessions by the user gave five different profiles, and that is acceptable for a quiz, as with personality tests. The claim that the method learns a known preference rests on the benchmark with simulated users. Frame results as "your style in this session".
- [x] Profile table on the test page: "no clear preference" under 75%, a lean up to 95%, stated plainly above; plus a 12-round option.
- [ ] Run a QLoRA comparison: the same sessions with the model loaded in 4-bit and at full precision, reporting memory used, seconds per round, and any change in results.
- [ ] Add an "axes right" summary to `bench/report.py` so analyses stop living in one-off scripts.

## Style menu

- [ ] Watch for overlap between casual and warm, and between collaborative and "explains why".
- [ ] Collaborative answers read as if they give reasons even under "just the advice": "plain statements about what works" comes out as "X can help you Y", which is a reason in all but name. Seen in a live session on 2026-11-25 (profile said just the advice, the answer explained). Depth was also the scale the judge agreed with least (15 of 22). Either accept it and say so, or find a wording for collaborative that states what people do without saying what it achieves.
- [ ] Deal with warm and "explains why" answers running past three sentences before the cap.
- [ ] Remove the retired axes (brevity, language, energy) and old grids from `data/advice.json` and `bench/remote.py`.
- [ ] Write prepared question sets with checked fact sheets for doctor and tutor: the fallback when Gemini is down, the example chips, and settings the benchmark can be rerun on. (The built-in ten mix settings, which may be one reason the user's own sessions differed.)

## Real app

- [x] Backend (`app/backend`): password check, signed session cookie, and a private relay to Modal. Runs locally; a Dockerfile for Railway exists but is not deployed or tested there.
- [x] Front end (`app/frontend`), first version: sign-in, welcome, questions, profile with four scales, before and after, round-by-round charts.
- [ ] Serve the page from Railway so it loads instantly while the GPU starts; add loading states for the cold start.
- [x] (Done: the try box.) "Try your model" on the results screen: the visitor types any text or question and sees the standard model's version beside their own model's, as with the prepared examples. This is the in-app form of converting text. Needs one endpoint that runs both models while the session's GPU is still up (it shuts down five minutes after last use, so the box must say when it has expired), a decision on prompt wording for rewording against answering, and a length limit on the input.
- [x] Scales measured from the writing, the same measure everywhere (built 2026-11-25): after every round the tuned model answers the three unseen questions and the judge rates each scale 1 to 5; charts and profile plot the average, with the style-choice measure as fallback when the judge is down.
- [x] Training charts removed from the app (2026-11-25, user's call): the model is rated once at the end, on six unseen questions, and each scale shows the standard model's marker beside the tuned model's. Per-round measuring made waits longer and the charts were noisy with five points. The benchmark still records every round.
- [x] (Superseded: live sessions use 2e-5 with the choice clipped.) At learning rate 3e-5 the live model's style choice commits very hard in one round (warmth log-odds of -24 after the first question in a scripted session). The writing changes more slowly than the choice. Revisit whether 3e-5 is right for the 5-question session now that the page shows the writing.
- [x] Experiment: credit the style an answer displays, not the one it was assigned (`--relabel`). Not adopted: 12 runs against the matching 12 without it gave 3.50 against 3.67 axes, a 51% answer-level win rate, and no better agreement between choice and writing (3.50 against 3.67 of 4). About one answer in four was relabelled per round. The flag stays in the benchmark, off.
- [x] Wait indicators: good enough as built (2026-11-25). Every wait names what it is for (drafting questions, starting the session, training on a pick, building results, packaging the model), with a spinner and, at start-up, the steps and elapsed time. Not doing: reporting which exact step the server is on, which would need a status endpoint and polling for a wait of under a minute.
- [ ] Enforce the session time limit on the server.
- [x] (Done in its current form: summary, four scales rated from the writing, examples; per-round charts dropped by choice.) Results screen: present the outcome as a profile, show the user's style against its opposite, keep the per-round charts and tracked answers.
- [x] (Done: footer, try box and model card, 2026-11-26.) Add a visible note that the answers are a style demonstration, not advice.
- [x] Dynamic settings and questions (built 2026-11-25): the visitor types who the model should speak as; Gemini checks it and drafts the questions with two-point fact sheets (bare advice, no reasons); the visitor can edit any question, and edited ones get a fact sheet at the start; the six rating questions come from the same setting; the setting goes into the downloaded package; "general everyday questions" remains as the built-in option. `bench/setting.py`, `/prepare` on the Modal CPU service.
- [x] Decided (2026-11-26): no fallback to built-in questions when Gemini is down. Three Gemini models are tried in turn, and the results rating needs Gemini too, so built-in questions would not save the session.
- [ ] Dynamic settings, still open: rewrite the built-in fact sheets without reasons; the first `prepare` after idle took 16 s (cold start), later ones about 5 s; the keyword content check is only approximate for generated questions (keywords are derived from each point).
- [ ] The standard model still rates as "explains why" (4.7 of 5) on generated fact sheets with no reasons in them: it adds reasons on its own. So that starting position is the model's habit, not the sheets'.
- [ ] Live content check by the judge, as a filter (never a rewrite): optional, not required. Measured 2026-11-25 (8 runs each, gentler update, judge as user): with the keyword check off, shown answers missing a fact rose from about 7% to about 14%, learning was unchanged (3.62 against 3.50 axes right) and content did not collapse. Not tested: a user who prefers shorter answers, which is where dropping content would be rewarded. Gemini took 6 to 7 s per call that day.
- [ ] About 30% of answers "add something" by the judge's standard, before and after training alike. Decide how strict that standard should be when hand-labelling.
- [ ] Decide what counts as a question the app accepts (requests for advice; the style menu does not fit trivia or creative writing) and what it declines.
- [ ] Check that generated fact sheets are good enough: short, two distinct points, no risky specifics.
- [x] Adapter download: packaged on Modal after the last round (adapter, profile, run script), kept for a day.
- [ ] Try the downloaded package on a real machine: run `example.py` and check the output matches the session.
- Rule: never deploy while a session is in progress; a deploy drops it.

## Scope decisions (2026-11-25)

- The app offers one session length, five questions. Ten is still accepted by the GPU service for experiments.
- Built 2026-11-25: the "try your model" box on the results screen (type a question, see the standard model's answer beside your model's). No LLM commentary on the difference. The finished session stays loaded until it has been idle for half an hour or the GPU shuts down (five minutes after last use).
- Not planned: uploading a saved model, continuing training across sessions, and the standalone convert mode. The downloadable model stays.

## Advanced options (later)

- [ ] Let visitors continue from a saved adapter: an upload (safetensors only, validated) or a server-side resume code.
- [ ] Let visitors choose the number of rounds, within tested limits.
- [ ] Let visitors choose which axes to tune. With fewer styles than answers per round, sampling must allow repeats.
- [ ] A "convert" mode that skips the quiz: upload a saved adapter, paste text, and get it reworded straight away. Depends on the adapter download and upload items above, and on the file recording its base model, menu version and prompt setup.
- [ ] Later, if convert gets used: saved per-setting profiles and a content check that works on pasted text.

## Write-up and resume

- [ ] Benchmark page: GRPO against DPO, what failed and why (reward-model GRPO, DPO collapse), and how this departs from textbook GRPO.
- [x] (Done: 91% over base, 57% head-to-head, 51% fewer additions.) Fill the resume bullets from real runs; never name a baseline without a number against it.
- [ ] Settle the title: "Attune — A Real-Time RLHF Platform for Fine-Tuning an LLM to a User's Communication Style" ("Preferred" optional; "Platform" is fair once the app is live).
- [x] Expand the README before making the repo public (screenshots, architecture, results; 2026-11-26).

## Tooling

- [ ] Fix `bench.remote status`: it prints "waiting" for jobs that were never launched as well as for queued ones.
- [ ] Fix the `launch` message that reports not-yet-launched jobs as "already done".
- [ ] After a deploy, wait for old containers to stop before launching or testing; requests sent straight away land on stale code.
