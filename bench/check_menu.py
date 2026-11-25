"""Does the base model follow each menu option, within the length cap, without losing facts?

For every axis and option, generate answers with that option fixed and the other
axes varied, then report the axis's own style feature and fact retention.

    python -m bench.check_menu --scenario data/advice.json
"""
import argparse
import json
import random
from pathlib import Path

from . import run as R
from .personas import facts_retained, features
from .policy import Policy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--scenario", default="data/advice.json")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--per-option", type=int, default=24)
    ap.add_argument("--axes", default=None, help="comma-separated subset of the menu's axes (default: all)")
    ap.add_argument("--solo", action="store_true", help="test each axis on its own, one clause per style line")
    ap.add_argument("--position", choices=["line", "user", "user_bullets"], default="line",
                    help="line: the model's own style line; user: instruction at the end of the user turn; "
                         "user_bullets: the same, one bullet per axis")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    scenario = json.loads(Path(args.scenario).read_text())
    menu = scenario["menu"]
    if args.axes:
        menu = [m for name in args.axes.split(",") for m in menu if m["axis"] == name]
    rng = random.Random(args.seed)
    policy = Policy(args.model)
    system = scenario["system"] + R.STYLE_FORMAT

    report = {"axes": {}, "samples": []}
    for j, m in enumerate(menu):
        R.use_menu(menu, [m["axis"]] if args.solo else [x["axis"] for x in menu])
        report["axes"][m["axis"]] = {}
        for first in (True, False):
            name = m["options"][0 if first else 1][0]
            styles = [s for s, on in zip(R.STYLES, R.AXES[m["axis"]]) if on == first]
            picks = [(rng.choice(scenario["train"]), rng.choice(styles)) for _ in range(args.per_option)]
            if args.position == "line":
                prompts = [policy.prompt_ids(system, R.user_message(it))
                           + policy.tok(f"Style: {s}\n\n", add_special_tokens=False).input_ids for it, s in picks]
            else:
                def ask(it, s):
                    if args.position == "user_bullets":
                        clauses = [o[1] for x in menu for o in x["options"] if o[1] in s]
                        s = "\n" + "\n".join(f"- {c}" for c in clauses)
                    return policy.prompt_ids(scenario["system"],
                                             f"{R.user_message(it)}\n\nWrite your answer in this style: {s}")
                prompts = [ask(it, s) for it, s in picks]
            outs = policy.generate(prompts, max_new_tokens=args.max_new_tokens, temperature=0.8, top_p=0.95,
                                   use_adapter=False)
            texts = [policy.decode(o) for o in outs]
            feats = [features(t) for t in texts]
            n = len(texts)
            report["axes"][m["axis"]][name] = {
                "feature": m["feature"],
                "feature_mean": sum(f[m["feature"]] for f in feats) / n,
                "words": sum(f["words"] for f in feats) / n,
                "sentences": sum(f["sentences"] for f in feats) / n,
                "over_3_sentences": sum(f["sentences"] > 3 for f in feats) / n,
                "over_4_sentences": sum(f["sentences"] > 4 for f in feats) / n,
                "has_line_break": sum("\n" in t for t in texts) / n,
                # share of answers where some paragraph has more than two sentences
                "long_paragraph": sum(any(features(par)["sentences"] > 2 for par in t.split("\n") if par.strip())
                                      for t in texts) / n,
                "cut_off": sum(o[-1] != policy.eos_id for o in outs) / n,
                "facts": sum(facts_retained(t, it["keys"]) for t, (it, _) in zip(texts, picks)) / n,
            }
            report["samples"] += [{"axis": m["axis"], "option": name, "style": s, "q": it["q"], "text": t}
                                  for (it, s), t in list(zip(picks, texts))[:6]]
    return report


if __name__ == "__main__":
    print(json.dumps(main()["axes"], indent=1))
