"""Launch benchmark grids on the deployed Modal app and collect their results.

    python -m bench.remote launch final4     # spawn every run not yet finished; returns at once
    python -m bench.remote status final4     # list progress and download finished results
    python -m bench.remote cancel final4     # stop the runs launched by the last `launch`

Finished results are downloaded to results/modal/<grid>/, ready for bench.report.
"""
import json
import sys
from pathlib import Path

import modal

# Short sessions on the advice scenario: two-step GRPO, four-sentence cap, six rounds.
# (Grids run before 2026-11-25 afternoon used three sentences and 96 tokens.)
LEARN = ["--method", "grpo_style", "--two-step", "--scenario", "data/advice.json", "--rounds", "6",
         "--max-new-tokens", "128", "--max-sentences", "4", "--min-facts", "0.5", "--eval-split", "test",
         "--lr", "3e-5"]
STRICT = ["--min-facts", "1.0", "--strict-facts"]
SEVEN_B = ["--model", "Qwen/Qwen2.5-7B-Instruct"]


# The live setup: 7B, four axes, best-and-worst feedback, neutral start, both facts required.
FOUR = "warmth,directiveness,formality,depth"
LIVE = LEARN + STRICT + SEVEN_B + ["--feedback", "bestworst", "--neutral-start", "--axes", FOUR]
LIVE_PERSONAS = ["gentle_explainer", "blunt_friend", "kind_professional", "reserved_analyst"]
# Ways to make the update gentler. Ten rounds, with the user's preferences reversing after
# round five: the first half shows how fast it learns, the second whether it can turn around.
GENTLE = {"base": [], "lr1e-5": ["--lr", "1e-5"],
          "cclip0.2": ["--choice-clip", "0.2"], "cclip0.5": ["--choice-clip", "0.5"],
          "cclip0.2_lr1e-5": ["--choice-clip", "0.2", "--lr", "1e-5"]}


# The method the live app runs, and the two scales each simulated user cares about in the "two" test.
SHIPPED = ["--choice-clip", "0.2", "--lr", "2e-5", "--answer-weight", "0.3", "--min-distance", "2", "--no-duplicates"]
CARE = {"gentle_explainer": "warmth,depth", "blunt_friend": "directiveness,formality",
        "kind_professional": "warmth,formality", "reserved_analyst": "directiveness,depth"}


def gentle(variants, seeds):
    return [{"name": f"{tag}_{p}_s{s}",
             "argv": LIVE + ["--rounds", "10", "--flip-at", "5", "--persona", p, "--seed", str(s), *GENTLE[tag]]}
            for tag in variants for p in LIVE_PERSONAS for s in seeds]


GRIDS = {
    "gentle_smoke": [{"name": "gentle_smoke",
                      "argv": LIVE + ["--rounds", "2", "--flip-at", "1", "--heldout", "3", "--choice-clip", "0.2",
                                      "--persona", "gentle_explainer"]}],
    "gentle": gentle(GENTLE, [1]),
    "judge_smoke": [{"name": "judge_smoke",
                     "argv": LIVE + ["--rounds", "2", "--flip-at", "1", "--flip-axes", "warmth,depth", "--heldout", "3",
                                     "--judge-user", *GENTLE["cclip0.2_lr1e-5"], "--persona", "gentle_explainer"]}],
    # The same comparison with the judge model as the simulated user (it can see formality, which
    # the word lists cannot), and a milder reversal: two of the four axes flip after round five.
    "gentle2": [{**j, "argv": j["argv"] + ["--judge-user", "--flip-axes", "warmth,depth"]}
                for j in gentle(["base", "cclip0.2_lr1e-5"], [1, 2])],
    "menu_check10": [{"name": "formality_7B", "module": "check_menu",   # formality in place of language
                      "argv": ["--position", "user_bullets", "--axes", "warmth,directiveness,formality,depth", *SEVEN_B]}],
    "menu_check11": [{"name": "formality_v2_7B", "module": "check_menu",   # casual without slang
                      "argv": ["--position", "user_bullets", "--axes", "warmth,directiveness,formality,depth", *SEVEN_B]}],
    "menu_check12": [{"name": "collaborative_v2_7B", "module": "check_menu",   # collaborative without the hedging
                      "argv": ["--position", "user_bullets", "--axes", "warmth,directiveness,formality,depth", *SEVEN_B]}],
    # The gentler update with the content check switched off: does content hold without it?
    # Compared with the same runs in gentle2, and scored afterwards by the judge.
    "nocheck": [{**j, "name": "nocheck_" + j["name"],
                 "argv": j["argv"] + ["--judge-user", "--flip-axes", "warmth,depth", "--min-facts", "0"]}
                for j in gentle(["cclip0.2_lr1e-5"], [1, 2])],
    "menu_check14": [{"name": "four_sentences_two_paragraphs_7B", "module": "check_menu",   # new length and layout rule
                      "argv": ["--position", "user_bullets", "--axes", FOUR, "--max-new-tokens", "128", *SEVEN_B]}],
    "menu_check15": [{"name": "one_point_three_sentences_7B", "module": "check_menu",   # one point per question
                      "argv": ["--position", "user_bullets", "--axes", FOUR, "--scenario", "data/experiments/advice_one.json", *SEVEN_B]}],
    # Quick GRPO-against-DPO check on the live setup: judge as the user, best-and-worst feedback
    # for both, five rounds, one seed. Scored afterwards with `python -m bench.judge compare quick`.
    "quick": [{"name": f"{tag}_{p}_s1",
               "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", "1", *extra]}
              for tag, extra in [("grpo", GENTLE["cclip0.2_lr1e-5"]),
                                 ("dpo-lr3e-5", ["--method", "dpo", "--lr", "3e-5"]),
                                 ("dpo-lr1e-5", ["--method", "dpo", "--lr", "1e-5"])]
              for p in LIVE_PERSONAS],
    # The full GRPO-against-DPO comparison: three seeds, four simulated users, each method at two
    # learning rates. Scored with `python -m bench.judge compare full`.
    "full": [{"name": f"{tag}_{p}_s{seed}",
              "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed), *extra]}
             for tag, extra in [("grpo-lr1e-5", ["--choice-clip", "0.2", "--lr", "1e-5"]),
                                ("grpo-lr3e-5", ["--choice-clip", "0.2", "--lr", "3e-5"]),
                                # every axis shows both options in each round, and no two answers are identical
                                ("grpo-spread-lr3e-5", ["--choice-clip", "0.2", "--lr", "3e-5", "--spread", "--no-duplicates"]),
                                # the reward goes to the style each answer displays (added 2026-11-25, evening)
                                ("grpo-relabel-lr3e-5", ["--choice-clip", "0.2", "--lr", "3e-5", "--relabel"]),
                                ("dpo-lr1e-5", ["--method", "dpo", "--lr", "1e-5"]),
                                ("dpo-lr3e-5", ["--method", "dpo", "--lr", "3e-5"])]
             for p in LIVE_PERSONAS for seed in (1, 2, 3)],
    # Three wordings of the warm option: the current one (an opening remark), a looser one, and one in between.
    "menu_check16": [{"name": f"warm_{tag}_7B", "module": "check_menu",
                      "argv": ["--position", "user_bullets", "--axes", FOUR, "--max-new-tokens", "128", "--save-all",
                               "--scenario", f, *SEVEN_B]}
                     for tag, f in [("current", "data/advice.json"), ("loose", "data/experiments/advice_warm_loose.json"),
                                    ("anywhere", "data/experiments/advice_warm_anywhere.json")]],
    # Mood-matching wordings of the warm option against the current one, with six good-news questions added
    # to the ten training questions: does the opening follow the mood, and is it less of a stock phrase?
    "menu_check18": [{"name": f"mood_{tag}_7B", "module": "check_menu",
                      "argv": ["--position", "user_bullets", "--axes", FOUR, "--max-new-tokens", "128", "--save-all",
                               "--per-option", "36", "--scenario", f"data/experiments/advice_mood_{tag}.json", *SEVEN_B]}
                     for tag in ("current", "feeling", "mood")],
    # Train the wording more gently than the style choice, to stop the model settling on one stock
    # phrasing or writing the same answer in every style (seen in live sessions at 3e-5).
    "wording": [{"name": f"{tag}_{p}_s{seed}",
                 "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed),
                                 "--choice-clip", "0.2", "--lr", "3e-5", *extra]}
                for tag, extra in [("grpo-lr2e-5", ["--lr", "2e-5"]),   # a learning rate between the two tested
                                   # the live setting from 2026-11-25 late evening
                                   ("grpo-lr2e-5-aw0.3", ["--lr", "2e-5", "--answer-weight", "0.3"]),
                                   ("grpo-aw0.3", ["--answer-weight", "0.3"]),
                                   ("grpo-aw0.3-spread", ["--answer-weight", "0.3", "--spread", "--no-duplicates"])]
                for p in LIVE_PERSONAS for seed in (1, 2, 3)],
    # The frozen method (2026-11-25, late): what the live app runs, against DPO at two learning rates.
    # These are the numbers to quote. Score with `python -m bench.judge compare final`.
    "final": [{"name": f"{tag}_{p}_s{seed}",
               "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed), *extra]}
              for tag, extra in [("grpo", ["--choice-clip", "0.2", "--lr", "2e-5", "--answer-weight", "0.3",
                                           "--min-distance", "2", "--no-duplicates"]),
                                 ("dpo-lr2e-5", ["--method", "dpo", "--lr", "2e-5"]),
                                 ("dpo-lr3e-5", ["--method", "dpo", "--lr", "3e-5"])]
              for p in LIVE_PERSONAS for seed in (1, 2, 3)],
    # Less tidy simulated users: one who cares about two scales only, one who picks at random a quarter
    # of the time, and one who picks on length alone. Score with `python -m bench.judge users`.
    "users": [{"name": f"{kind}-{method}_{p}_s{seed}",
               "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed), *how(p), *extra]}
              for kind, how in [("two", lambda p: ["--care", CARE[p]]),
                                ("noisy", lambda p: ["--pick-noise", "0.25"]),
                                ("short", lambda p: ["--prefer-short"])]
              for method, extra in [("grpo", SHIPPED), ("dpo", ["--method", "dpo", "--lr", "2e-5"])]
              for p in LIVE_PERSONAS for seed in (1, 2)],
    # Rerun of the matched pairs after the sentence-cap fix (2026-11-26): styled answers had been cut at
    # three sentences while plain ones (the standard model's, DPO's at evaluation) could use four.
    # Score with `python -m bench.judge compare final4` and `python -m bench.judge users users4`.
    # Repeated phrasing in the tuned model's answers: how strongly the wording is trained (0.3 is shipped; 0 trains
    # the style choice only), with the final answers also sampled at several temperatures by the tuned model and
    # by the untouched one in the same style.
    "sameness": [{"name": f"aw{aw}_{p}_s{seed}",
                  "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed), "--max-sentences", "4",
                                  *SHIPPED[:4], "--answer-weight", aw, *SHIPPED[6:], "--sample-temps", "0,0.7,1.0"]}
                 for aw in ("0.3", "0.15", "0") for p in LIVE_PERSONAS for seed in (1, 2, 3)],
    # Can what GRPO learned be moved into the plain prompt, so the model writes in the learned style with no
    # style in its prompt? After the session, the adapter is trained to give its own styled answers to the
    # plain prompt, at three settings; the held-out questions are then answered from the plain prompt.
    "distill": [{"name": f"grpo_{p}_s{seed}",
                 "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed), *SHIPPED,
                                 "--distill", "3:1e-4,6:1e-4,3:3e-4"]}
                for p in LIVE_PERSONAS for seed in (1, 2)],
    # GRPO's update aimed at the plain prompt (no style choice, no style in the tuned model's prompt), at two
    # learning rates. Compared with the GRPO and DPO runs of final4, which use the same users and seeds.
    "plain": [{"name": f"grpo-plain-lr{lr}_{p}_s{seed}",
               "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed),
                               "--method", "grpo_plain", "--lr", lr]}
              for lr in ("2e-5", "5e-5") for p in LIVE_PERSONAS for seed in (1, 2, 3)],
    # Fair tuning of the two methods that train the plain prompt: the same three learning rates, users and seeds,
    # both reusing earlier rounds' picks (DPO always does). Each run also answers the five held-out questions:
    # a method's rate is chosen on those and the comparison reported on the ten test questions.
    "tune1": [{"name": f"{tag}-lr{lr}_{p}_s{seed}",
               "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed), "--save-probe",
                               "--lr", lr, *extra]}
              for tag, extra in [("dpo", ["--method", "dpo"]), ("grpo-plain", ["--method", "grpo_plain", "--replay"])]
              for lr in ("2e-5", "5e-5", "1e-4") for p in LIVE_PERSONAS for seed in (1, 2, 3)],
    # Session length and the model's own answers, for the two methods that train the plain prompt. Ten rounds
    # (a ten-round run passes through every shorter session, so rounds 5, 7 and 8 are read from the same runs),
    # with 0, 1 or 2 of each round's four answers being the model's own answer to the plain prompt. Rate 2e-5
    # for both: in tune1 it was the highest rate at which five rounds lost no content.
    "long": [{"name": f"{tag}-own{own}-lr2e-5_{p}_s{seed}",
              "argv": LIVE + ["--judge-user", "--rounds", "10", "--persona", p, "--seed", str(seed), "--save-probe",
                              "--lr", "2e-5", "--own", str(own), "--no-duplicates", *extra]}
             for tag, extra in [("dpo", ["--method", "dpo"]), ("grpo-plain", ["--method", "grpo_plain", "--replay"])]
             for own in (0, 1, 2) for p in LIVE_PERSONAS for seed in (1, 2, 3)],
    # The model's own answers at five rounds and rate 5e-5, the best five-round rate in tune1 (the "long" grid
    # covers five rounds only at 2e-5). Own 0 is the control: the same runs with the duplicate rule on.
    "own5": [{"name": f"{tag}-own{own}-lr5e-5_{p}_s{seed}",
              "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed), "--save-probe",
                              "--lr", "5e-5", "--own", str(own), "--no-duplicates", *extra]}
             for tag, extra in [("dpo", ["--method", "dpo"]), ("grpo-plain", ["--method", "grpo_plain", "--replay"])]
             for own in (0, 1, 2) for p in LIVE_PERSONAS for seed in (1, 2, 3)],
    # The "long" grid's best setup (no own answers, rate 2e-5) again on seeds it has not seen, so the session
    # length chosen from "long" can be reported on runs that played no part in choosing it.
    "confirm": [{"name": f"{tag}-own0-lr2e-5_{p}_s{seed}",
                 "argv": LIVE + ["--judge-user", "--rounds", "10", "--persona", p, "--seed", str(seed), "--save-probe",
                                 "--lr", "2e-5", "--own", "0", "--no-duplicates", *extra]}
                for tag, extra in [("dpo", ["--method", "dpo"]), ("grpo-plain", ["--method", "grpo_plain", "--replay"])]
                for p in LIVE_PERSONAS for seed in (4, 5, 6)],
    # What reusing earlier rounds' picks adds, at the shipped setup: the "confirm" GRPO runs again with it off.
    "ablate": [{"name": f"grpo-noreplay-own0-lr2e-5_{p}_s{seed}",
                "argv": LIVE + ["--judge-user", "--rounds", "10", "--persona", p, "--seed", str(seed), "--save-probe",
                                "--lr", "2e-5", "--own", "0", "--no-duplicates", "--method", "grpo_plain"]}
               for p in LIVE_PERSONAS for seed in (4, 5, 6)],
    "final4": [{"name": f"{tag}_{p}_s{seed}",
                "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed), *extra]}
               for tag, extra in [("grpo", SHIPPED), ("dpo-lr2e-5", ["--method", "dpo", "--lr", "2e-5"])]
               for p in LIVE_PERSONAS for seed in (1, 2, 3)],
    "users4": [{"name": f"{kind}-{method}_{p}_s{seed}",
                "argv": LIVE + ["--judge-user", "--rounds", "5", "--persona", p, "--seed", str(seed), *how(p), *extra]}
               for kind, how in [("two", lambda p: ["--care", CARE[p]]),
                                 ("noisy", lambda p: ["--pick-noise", "0.25"])]
               for method, extra in [("grpo", SHIPPED), ("dpo", ["--method", "dpo", "--lr", "2e-5"])]
               for p in LIVE_PERSONAS for seed in (1, 2)],
}

volume = modal.Volume.from_name("attune-results")


def remote_files(name):
    try:
        return {Path(e.path).name for e in volume.listdir(name)}
    except Exception:
        return set()


def failure(call_id):
    """Short error text if this call has already failed, else None."""
    if not call_id:
        return None
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
    except TimeoutError:
        return None  # still running or queued
    except BaseException as e:  # the remote error, e.g. SystemExit from argparse
        return (str(e) or type(e).__name__)[:120]
    return None


def launch(name, limit=None):
    done = remote_files(name)
    local = Path("results/modal") / name
    prior = json.loads((local / "calls.json").read_text()) if (local / "calls.json").exists() else {}
    todo = [j for j in GRIDS[name] if f"{j['name']}.json" not in done
            and (j["name"] not in prior or failure(prior[j["name"]]))]
    todo = todo[:limit]  # launch in small batches when asked
    calls = {j["name"]: modal.Function.from_name("attune-bench", j.get("fn", "run_one"))
             .spawn({**j, "grid": name}).object_id for j in todo}
    local.mkdir(parents=True, exist_ok=True)
    old = json.loads((local / "calls.json").read_text()) if (local / "calls.json").exists() else {}
    (local / "calls.json").write_text(json.dumps({**old, **calls}, indent=1))
    finished = sum(f"{j['name']}.json" in done for j in GRIDS[name])
    print(f"{name}: launched {len(todo)} of {len(GRIDS[name])} runs "
          f"({finished} finished, {len(GRIDS[name]) - finished - len(todo)} already launched or held back)")


def status(name):
    files = remote_files(name)
    local = Path("results/modal") / name
    local.mkdir(parents=True, exist_ok=True)
    done = running = failed = queued = 0
    calls_file = local / "calls.json"
    calls = json.loads(calls_file.read_text()) if calls_file.exists() else {}
    for j in GRIDS[name]:
        n = j["name"]
        if f"{n}.json" in files:
            done += 1
            target = local / f"{n}.json"
            if not target.exists():
                target.write_bytes(b"".join(volume.read_file(f"{name}/{n}.json")))
            d = json.loads(target.read_text())
            if "rounds" not in d:
                print(f"  done     {n:40} (report, {d['wall_seconds']:.0f}s)")
                continue
            last = d["rounds"][-1]
            print(f"  done     {n:40} score {d['base']['score']:.2f} -> {last['score']:.2f} | "
                  f"win {last['win_rate']:.2f} | facts {last['facts']:.2f} | {d['wall_seconds']:.0f}s")
        elif failure(calls.get(n)):
            failed += 1
            print(f"  FAILED   {n}: {failure(calls.get(n))}")
        elif f"{n}.started" in files:
            running += 1
            print(f"  running  {n}")
        elif n in calls:
            queued += 1
            print(f"  queued   {n}")
        else:
            print(f"  not launched  {n}")
    print(f"{name}: {done} done, {running} running, {failed} failed, {queued} queued, "
          f"{len(GRIDS[name]) - done - running - failed - queued} not launched")


def cancel(name):
    calls = json.loads((Path("results/modal") / name / "calls.json").read_text())
    for n, call_id in calls.items():
        modal.FunctionCall.from_id(call_id).cancel()
    print(f"{name}: cancelled {len(calls)} calls")


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = None
    if "--limit" in args:  # e.g. launch final4 --limit 4
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]
    action, names = args[0], args[1:] or list(GRIDS)
    for name in names:
        if action == "launch":
            launch(name, limit)
        else:
            {"status": status, "cancel": cancel}[action](name)
