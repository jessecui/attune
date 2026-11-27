"""Score a grid of plain-prompt runs (DPO and plain-prompt GRPO) with the judge, after several session lengths.

A run of N rounds passes through every shorter session on the way, so its answers after round n are a
session of n rounds. Each method's answers to the ten test questions are rated for how well they suit
the simulated user, whether they keep the fact sheet, and whether they add to it. The judge does not
look at whether the text is well formed, so that is checked here by pattern (see malformed).

    python -m bench.score confirm 8          # the numbers quoted in the README
    python -m bench.score long 5 7 8 10      # session length, and the model's own answers among the four

Verdicts are kept in results/modal/<grid>_verdicts.json, so a rerun only judges what is new.
"""
import glob
import os
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from . import run as R
from .judge import judge_many, style_score

GRID = sys.argv[1]
LENGTHS = [int(x) for x in sys.argv[2:]] or [8]
sc = json.loads(Path("data/advice.json").read_text())
menu = sc["menu"]
facts = {it["q"]: it["facts"] for it in sc["test"]}
test_q = [it["q"] for it in sc["test"]]
R.use_menu(menu, "warmth,directiveness,formality,depth".split(","))

runs = defaultdict(dict)   # (method, own) -> (persona, seed) -> {rounds: texts, 0: base texts}
for f in glob.glob(f"results/modal/{GRID}/*.json"):
    d = json.loads(Path(f).read_text())
    if "rounds" not in d:
        continue
    parts = d["name"].split("_")
    tag, key = parts[0], ("_".join(parts[1:-1]), parts[-1])
    method, rest = tag.split("-own")
    runs[method, int(rest.split("-lr")[0])][key] = {0: d["base"]["texts"], **{r["round"]: r["texts"] for r in d["rounds"]}}

cache_file = Path("results/modal") / f"{GRID}_verdicts.json"
cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
h = lambda q, t: hashlib.sha1(f"{q}|{t}".encode()).hexdigest()
todo = {h(q, t): (q, facts[q], t) for by_run in runs.values() for r in by_run.values() for n in [0, *LENGTHS] if n in r
        for q, t in zip(test_q, r[n]) if h(q, t) not in cache}
print(f"judging {len(todo)} answers ({len(cache)} cached)")
keys = list(todo)
for i in range(0, len(keys), 240):
    part = keys[i:i + 240]
    for k, v in zip(part, judge_many([todo[k] for k in part], workers=int(os.environ.get('WORKERS', 8)))):
        cache[k] = v
    cache_file.write_text(json.dumps(cache))

mean = lambda v: sum(v) / len(v)


# Words the system word list lacks but that are ordinary English, so the spelling check passes them.
MODERN = set("""prioritize prioritizing prioritized prioritizes email emails roadmap benchmark benchmarks proactive proactively
mindset doesn't aren't isn't healthcare apps paid impactful online creditworthiness meds reusable confrontational okay stats info
meetups automate feet overhydration checklist mimicking behaviours timeframe brainstorming paperwork podcast backpack escalations
moisturized sustainability expertise hashtag timelines upfront miscommunications bullseye demystify judgmental breadcrumbs
budgetable papertrail""".split())
WORD_LIST = Path("/usr/share/dict/words")   # on macOS and most Linux systems; without it only the patterns are checked
WORDS = {w.strip().lower() for w in WORD_LIST.read_text().split("\n")} if WORD_LIST.exists() else None
if WORDS is None:
    print(f"no word list at {WORD_LIST}: misspellings are not counted, so the malformed column is a lower bound")
tokens = lambda text: re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("’", "'"))
# Every word the untouched model wrote, and every word of the questions and fact sheets, counts as spelled right.
SEEN = {w for by_run in runs.values() for r in by_run.values() for t in r[0] for w in tokens(t)} | {
    w for split in ("train", "heldout", "test") for it in sc[split] for w in tokens(it["q"] + " " + " ".join(it["facts"]))}
ENDINGS = (("'s", ""), ("s", ""), ("es", ""), ("ies", "y"), ("ed", ""), ("ed", "e"), ("d", ""), ("ing", ""), ("ing", "e"), ("ly", ""),
           ("er", ""), ("est", ""), ("ier", "y"), ("iest", "y"), ("ily", "y"), ("ied", "y"))


def spelled(w):
    """Whether a word is in the word list, in a plain or inflected form (planned, planning, plans)."""
    if w in WORDS or w in SEEN or w in MODERN or len(w) <= 3:
        return True
    if any(w.endswith(end) and w[: -len(end)] + to in WORDS for end, to in ENDINGS):
        return True
    return any(w.endswith(end) and len(w) > len(end) + 2 and w[-len(end) - 1] == w[-len(end) - 2] and w[: -len(end) - 1] in WORDS
               for end in ("ing", "ed", "er"))


def malformed(text):
    """Broken text of the kinds over-training produces: no space after punctuation, two words run
    together, an answer that does not end, characters from another script, or a misspelled word
    ("nerve-wrking"). The judge sees none of these."""
    if (re.search(r"[a-z][.,;:!?][A-Za-z]", text) or re.search(r"\b[a-z]{2,}[A-Z][a-z]+", text)
            or not re.search(r"[.!?)\"']\s*$", text.strip()) or re.search(r"[^\x00-\x7f‘’“”–—…é°]", text)):
        return True
    return WORDS is not None and not all(spelled(w) or spelled(w.split("'")[0]) for w in tokens(text))


def fits(texts, key):
    wants = R.user_wants(key[0], menu)
    return [style_score(cache[h(q, t)], wants) for q, t in zip(test_q, texts)]


def right(texts, key):
    wants = R.user_wants(key[0], menu)
    vs = [cache[h(q, t)] for q, t in zip(test_q, texts)]
    return sum((mean([v[a] for v in vs]) > 3) == (w > 0) for a, w in wants.items())


def versus(a, na, b, nb):
    w = t = l = 0
    for key in a:
        if key in b and na in a[key] and nb in b[key]:
            for x, y in zip(fits(a[key][na], key), fits(b[key][nb], key)):
                w += x > y; t += x == y; l += x < y
    n = w + t + l
    return f"{100 * (w + t / 2) / n:.0f}% ({w}/{t}/{l})" if n else "none shared"


print("\nmethod | own answers per round | rounds | runs | fit | scales right of 4 (mean, variance) | facts | additions | malformed | over untouched | GRPO against DPO, same setup")
for (method, own), by_run in sorted(runs.items()):
    for n in LENGTHS:
        have = {k: r for k, r in by_run.items() if n in r}
        if not have:
            continue
        vs = [cache[h(q, t)] for r in have.values() for q, t in zip(test_q, r[n])]
        rr = [right(r[n], k) for k, r in have.items()]
        other = runs.get(("dpo", own)) if method == "grpo-plain" else None
        print(f"  {method:10} | {own} | {n:2} | {len(have):2} | {mean([f for k, r in have.items() for f in fits(r[n], k)]):5.2f} | "
              f"{mean(rr):.2f}  {statistics.pvariance(rr):.2f} | {mean([mean(v['facts_covered']) for v in vs]):.2f} | "
              f"{mean([bool(v['added']) for v in vs]):.2f} | {100 * mean([malformed(t) for r in have.values() for t in r[n]]):.1f}% | "
              f"{versus(have, n, have, 0):>18} | {versus(have, n, other, n) if other else ''}")
    print()
