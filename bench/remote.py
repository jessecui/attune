"""Launch benchmark grids on the deployed Modal app and collect their results.

    python -m bench.remote launch main       # spawn every run not yet finished; returns at once
    python -m bench.remote status main       # list progress and download finished results
    python -m bench.remote cancel main       # stop the runs launched by the last `launch`

Finished results are downloaded to results/modal/<grid>/, ready for bench.report.
"""
import json
import sys
from pathlib import Path

import modal

PERSONAS = ["warm_brief", "thorough_formal"]
SEEDS = [1, 2, 3]


def grid(prefix, methods, extra=()):
    return [{"name": f"{prefix}_{m}_{p}_s{s}",
             "argv": ["--method", m, "--persona", p, "--seed", str(s), "--lr", "3e-5",
                      "--eval-split", "test", *extra]}
            for m in methods for p in PERSONAS for s in SEEDS]


# Short sessions on the advice scenario: two-step GRPO, four-sentence cap, six rounds.
# (Grids run before 2026-11-25 afternoon used three sentences and 96 tokens.)
LEARN = ["--method", "grpo_style", "--two-step", "--scenario", "data/advice.json", "--rounds", "6",
         "--max-new-tokens", "128", "--max-sentences", "4", "--min-facts", "0.5", "--eval-split", "test",
         "--lr", "3e-5"]
A3 = "warmth,directiveness,language"
A4 = A3 + ",energy"
A5 = "brevity," + A4
NEW_PERSONAS = ["gentle_guide", "brisk_expert", "upbeat_coach"]


STRICT = ["--min-facts", "1.0", "--strict-facts"]
SEVEN_B = ["--model", "Qwen/Qwen2.5-7B-Instruct"]


def learn(tag, axes, extra=()):
    return [{"name": f"{tag}_{p}_s{s}", "argv": LEARN + ["--axes", axes, "--persona", p, "--seed", str(s), *extra]}
            for p in NEW_PERSONAS for s in (1, 2)]


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
    "smoke": [{"name": "smoke", "argv": ["--method", "grpo_style", "--rounds", "2", "--heldout", "3"]}],
    "smoke_guard": [{"name": "smoke_guard", "argv": ["--method", "dpo", "--rounds", "2", "--eval-split", "test",
                                                     "--heldout", "3", "--guard", "-1"]}],  # always rolls back
    "menu_check": [{"name": "menu_check", "module": "check_menu", "argv": ["--scenario", "data/advice.json"]}],
    "menu_check2": [
        {"name": "solo_1.5B", "module": "check_menu", "argv": ["--solo"]},
        {"name": "three_axes_1.5B", "module": "check_menu", "argv": ["--axes", "warmth,directiveness,brevity"]},
        {"name": "five_axes_3B", "module": "check_menu", "argv": ["--model", "Qwen/Qwen2.5-3B-Instruct"]},
        {"name": "five_axes_7B", "module": "check_menu", "argv": ["--model", "Qwen/Qwen2.5-7B-Instruct"]},
    ],
    "menu_check3": [  # concrete option wording, looser system prompt
        {"name": "v2_five_axes_1.5B", "module": "check_menu", "argv": []},
        {"name": "v2_solo_1.5B", "module": "check_menu", "argv": ["--solo"]},
        {"name": "v2_five_axes_7B", "module": "check_menu", "argv": ["--model", "Qwen/Qwen2.5-7B-Instruct"]},
    ],
    "menu_check4": [  # where the instruction sits, and how many axes at once
        {"name": "user_five_1.5B", "module": "check_menu", "argv": ["--position", "user"]},
        {"name": "bullets_five_1.5B", "module": "check_menu", "argv": ["--position", "user_bullets"]},
        {"name": "bullets_five_7B", "module": "check_menu",
         "argv": ["--position", "user_bullets", "--model", "Qwen/Qwen2.5-7B-Instruct"]},
        {"name": "line_three_1.5B", "module": "check_menu", "argv": ["--axes", "warmth,directiveness,brevity"]},
        {"name": "line_two_1.5B", "module": "check_menu", "argv": ["--axes", "warmth,directiveness"]},
    ],
    "learn_smoke": [{"name": "learn_smoke", "argv": LEARN + ["--axes", A5, "--rounds", "2", "--heldout", "3",
                                                             "--persona", "upbeat_coach"]}],
    "learn": (learn("a3_1.5B", A3) + learn("a4_1.5B", A4) + learn("a5_1.5B", A5)
              + learn("a5_7B", A5, ["--model", "Qwen/Qwen2.5-7B-Instruct"])),
    # As "learn", but both facts required of every candidate and a cap that never cuts a fact.
    "learn2": (learn("v2_a4_1.5B", A4, STRICT) + learn("v2_a5_1.5B", A5, STRICT)
               + learn("v2_a5_7B", A5, STRICT + ["--model", "Qwen/Qwen2.5-7B-Instruct"])),
    "menu_check5": [  # third wording: no quotable examples
        {"name": "v3_bullets_five_7B", "module": "check_menu",
         "argv": ["--position", "user_bullets", "--model", "Qwen/Qwen2.5-7B-Instruct"]},
        {"name": "v3_bullets_five_1.5B", "module": "check_menu", "argv": ["--position", "user_bullets"]},
    ],
    # How the user gives feedback, on the 7B model with five axes.
    "feedback": (learn("rank4_7B", A5, STRICT + SEVEN_B)
                 + learn("bestworst4_7B", A5, STRICT + SEVEN_B + ["--feedback", "bestworst"])
                 + learn("pick2of6_7B", A5, STRICT + SEVEN_B + ["--k", "6", "--picks", "2"])),
    "menu_check6": [{"name": "v4_bullets_five_7B", "module": "check_menu",   # directiveness reworded again
                     "argv": ["--position", "user_bullets", "--model", "Qwen/Qwen2.5-7B-Instruct"]}],
    # One identical session per GPU type, to compare seconds per round and cost.
    "gpu_timing": [{"name": f"gpu_{g}", "fn": fn,
                    "argv": LEARN + ["--axes", A5, "--persona", "gentle_guide", "--seed", "1",
                                     "--feedback", "bestworst", *STRICT, *SEVEN_B]}
                   for g, fn in [("L4", "run_one"), ("A10", "run_one_a10"),
                                 ("L40S", "run_one_l40s"), ("A100", "run_one_a100")]],
    "menu_check7": [{"name": "v5_bullets_five_7B", "module": "check_menu",   # energy without "!" on everything
                     "argv": ["--position", "user_bullets", "--model", "Qwen/Qwen2.5-7B-Instruct"]}],
    # Does starting from "every style equally likely" learn as well as starting from the model's defaults?
    "neutral": (learn("default_start_7B", A5, STRICT + SEVEN_B + ["--feedback", "bestworst"])
                + learn("neutral_start_7B", A5, STRICT + SEVEN_B + ["--feedback", "bestworst", "--neutral-start"])),
    "menu_check8": [{"name": "v6_bullets_five_7B", "module": "check_menu",   # lighter energy wording
                     "argv": ["--position", "user_bullets", "--model", "Qwen/Qwen2.5-7B-Instruct"]}],
    "menu_check9": [   # depth in place of energy, with strict and with loose brevity
        {"name": "depth_strict_brevity_7B", "module": "check_menu",
         "argv": ["--position", "user_bullets", "--axes", "brevity,warmth,directiveness,language,depth", *SEVEN_B]},
        {"name": "depth_loose_brevity_7B", "module": "check_menu",
         "argv": ["--position", "user_bullets", "--scenario", "data/advice_loose.json", *SEVEN_B]},
        {"name": "depth_no_brevity_7B", "module": "check_menu",
         "argv": ["--position", "user_bullets", "--axes", "warmth,directiveness,language,depth", *SEVEN_B]},
    ],
    "menu_check10": [{"name": "formality_7B", "module": "check_menu",   # formality in place of language
                      "argv": ["--position", "user_bullets", "--axes", "warmth,directiveness,formality,depth", *SEVEN_B]}],
    "menu_check11": [{"name": "formality_v2_7B", "module": "check_menu",   # casual without slang
                      "argv": ["--position", "user_bullets", "--axes", "warmth,directiveness,formality,depth", *SEVEN_B]}],
    "menu_check12": [{"name": "collaborative_v2_7B", "module": "check_menu",   # collaborative without the hedging
                      "argv": ["--position", "user_bullets", "--axes", "warmth,directiveness,formality,depth", *SEVEN_B]}],
    "menu_check13": [   # two more wordings for collaborative: options without hedges, or statements not commands
        {"name": "collab_options_7B", "module": "check_menu",
         "argv": ["--position", "user_bullets", "--axes", "warmth,directiveness,formality,depth", *SEVEN_B]},
        {"name": "collab_statements_7B", "module": "check_menu",
         "argv": ["--position", "user_bullets", "--scenario", "data/advice_b.json",
                  "--axes", "warmth,directiveness,formality,depth", *SEVEN_B]},
    ],
    # The gentler update with the content check switched off: does content hold without it?
    # Compared with the same runs in gentle2, and scored afterwards by the judge.
    "nocheck": [{**j, "name": "nocheck_" + j["name"],
                 "argv": j["argv"] + ["--judge-user", "--flip-axes", "warmth,depth", "--min-facts", "0"]}
                for j in gentle(["cclip0.2_lr1e-5"], [1, 2])],
    "menu_check14": [{"name": "four_sentences_two_paragraphs_7B", "module": "check_menu",   # new length and layout rule
                      "argv": ["--position", "user_bullets", "--axes", FOUR, "--max-new-tokens", "128", *SEVEN_B]}],
    "menu_check15": [{"name": "one_point_three_sentences_7B", "module": "check_menu",   # one point per question
                      "argv": ["--position", "user_bullets", "--axes", FOUR, "--scenario", "data/advice_one.json", *SEVEN_B]}],
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
                     for tag, f in [("current", "data/advice.json"), ("loose", "data/advice_warm_loose.json"),
                                    ("anywhere", "data/advice_warm_anywhere.json")]],
    # Mood-matching wordings of the warm option against the current one, with six good-news questions added
    # to the ten training questions: does the opening follow the mood, and is it less of a stock phrase?
    "menu_check18": [{"name": f"mood_{tag}_7B", "module": "check_menu",
                      "argv": ["--position", "user_bullets", "--axes", FOUR, "--max-new-tokens", "128", "--save-all",
                               "--per-option", "36", "--scenario", f"data/advice_mood_{tag}.json", *SEVEN_B]}
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
    "main": grid("rank4", ["grpo_style", "dpo"]),
    "wide": grid("wide", ["grpo_style"], ["--menu", "wide"]),
    "pick8": grid("pick3of8", ["grpo_style"], ["--k", "8", "--picks", "3"]),
    "pick6": grid("pick2of6", ["grpo_style"], ["--k", "6", "--picks", "2"]),
    "dpo_pick8": grid("pick3of8", ["dpo"], ["--k", "8", "--picks", "3"]),
    "dpo_pick6": grid("pick2of6", ["dpo"], ["--k", "6", "--picks", "2"]),
    "dpo_guard": grid("guard", ["dpo"], ["--guard", "0.15"]),
    "dpo_beta": grid("beta05", ["dpo"], ["--beta", "0.5"]),
    "dpo_guard_pick8": grid("guard_pick3of8", ["dpo"], ["--k", "8", "--picks", "3", "--guard", "0.15"]),
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
    print(f"{name}: launched {len(todo)} of {len(GRIDS[name])} runs ({len(GRIDS[name]) - len(todo)} already done)")


def status(name):
    files = remote_files(name)
    local = Path("results/modal") / name
    local.mkdir(parents=True, exist_ok=True)
    done = running = failed = 0
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
        else:
            print(f"  waiting  {n}")
    print(f"{name}: {done} done, {running} running, {failed} failed, "
          f"{len(GRIDS[name]) - done - running - failed} waiting")


def cancel(name):
    calls = json.loads((Path("results/modal") / name / "calls.json").read_text())
    for n, call_id in calls.items():
        modal.FunctionCall.from_id(call_id).cancel()
    print(f"{name}: cancelled {len(calls)} calls")


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = None
    if "--limit" in args:  # e.g. launch feedback --limit 4
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]
    action, names = args[0], args[1:] or list(GRIDS)
    for name in names:
        if action == "launch":
            launch(name, limit)
        else:
            {"status": status, "cancel": cancel}[action](name)
