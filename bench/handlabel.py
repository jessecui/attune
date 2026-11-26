"""Check the judge model against a person's own ratings.

    python -m bench.handlabel make       # writes labels/handlabel.csv: 30 answers to rate, judge hidden
    python -m bench.handlabel compare    # after you fill it in: how often you and the judge agree

The answers come from the final benchmark and the less tidy users (their judge ratings are
already cached, so this costs nothing). The judge's ratings go to labels/handlabel_key.json;
don't open it until you've finished rating.

Scales, 1 to 5, with 3 for mixed or neither (the same ends the judge uses):
  warmth         5 = opens by recognising how the person may feel   1 = straight to the advice
  directiveness  5 = commands ("do this")                          1 = describes what works, leaves the choice
  formality      5 = relaxed, contractions                         1 = formal, no contractions
  depth          5 = a reason for each point                       1 = the advice only
  facts          how many of the two fact-sheet points the answer gives (0, 1 or 2)
  adds           y if it gives advice or claims not on the fact sheet, else n
"""
import csv
import json
import random
import sys
from pathlib import Path

from .judge import AXES

CACHES = ["results/judge/final_compare.json", "results/judge/users_compare.json"]
SHEET, KEY = Path("labels/handlabel.csv"), Path("labels/handlabel_key.json")
COLUMNS = ["id", "question", "fact_1", "fact_2", "answer", *AXES, "facts", "adds", "note"]


def _facts():
    sc = json.loads(Path("data/advice.json").read_text())
    return {it["q"]: it["facts"] for split in ("train", "heldout", "test") for it in sc[split]}


def make(n=30, seed=0):
    if SHEET.exists():
        sys.exit(f"{SHEET} already exists; delete it first if you want a fresh sample.")
    facts, pool = _facts(), {}
    for f in CACHES:
        pool.update(json.loads(Path(f).read_text()))
    pool = {k: v for k, v in pool.items() if k.split("|", 1)[0] in facts}
    # Draw so each axis has both ends represented, not mostly middling answers.
    rng = random.Random(seed)
    keys = sorted(pool)
    rng.shuffle(keys)
    picked, ends = [], {(a, hi): 0 for a in AXES for hi in (True, False)}
    for k in keys:
        v = pool[k]
        full = all(c >= 3 for c in ends.values())
        if len(picked) < n // 2 or full or any(ends[(a, v[a] >= 4)] < 3 and v[a] != 3 for a in AXES):
            picked.append(k)
            for a in AXES:
                if v[a] != 3:
                    ends[(a, v[a] >= 4)] += 1
        if len(picked) == n:
            break
    SHEET.parent.mkdir(exist_ok=True)
    with SHEET.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for i, k in enumerate(picked, 1):
            q, answer = k.split("|", 1)
            w.writerow([i, q, *facts[q], answer] + [""] * (len(COLUMNS) - 5))
    KEY.write_text(json.dumps({i: pool[k] for i, k in enumerate(picked, 1)}, indent=1))
    print(f"Wrote {len(picked)} answers to {SHEET}. Rate each in the empty columns, then run compare.")


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2
        i = j + 1
    return r


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return cov / den if den else float("nan")


def compare():
    key = {int(k): v for k, v in json.loads(KEY.read_text()).items()}
    rows = [r for r in csv.DictReader(SHEET.open()) if r["warmth"].strip()]
    if not rows:
        sys.exit(f"No ratings in {SHEET} yet.")
    side = lambda x: (x > 3) - (x < 3)
    print(f"{len(rows)} answers rated\n")
    print(f"  {'scale':14} same side | within 1 | exact | rank correlation")
    for a in AXES:
        mine = [int(r[a]) for r in rows]
        judge = [key[int(r["id"])][a] for r in rows]
        same = sum(side(x) == side(y) for x, y in zip(mine, judge)) / len(rows)
        near = sum(abs(x - y) <= 1 for x, y in zip(mine, judge)) / len(rows)
        exact = sum(x == y for x, y in zip(mine, judge)) / len(rows)
        print(f"  {a:14} {same:9.0%} | {near:8.0%} | {exact:5.0%} | {_spearman(mine, judge):.2f}")
    f = [r for r in rows if r["facts"].strip()]
    if f:
        agree = sum(int(r["facts"]) == sum(key[int(r["id"])]["facts_covered"]) for r in f) / len(f)
        print(f"\n  facts covered: same count in {agree:.0%} of {len(f)}")
    d = [r for r in rows if r["adds"].strip()]
    if d:
        pairs = [(r["adds"].strip().lower().startswith("y"), bool(key[int(r["id"])]["added"])) for r in d]
        agree = sum(x == y for x, y in pairs) / len(pairs)
        print(f"  adds something: agree in {agree:.0%} of {len(d)} (you said yes {sum(x for x, _ in pairs)}, "
              f"judge {sum(y for _, y in pairs)})")
    worst = sorted(rows, key=lambda r: -sum(abs(int(r[a]) - key[int(r["id"])][a]) for a in AXES))[:3]
    print("\n  largest disagreements (id: yours / judge's, in scale order):")
    for r in worst:
        k = key[int(r["id"])]
        print(f"    {r['id']}: {'-'.join(r[a] for a in AXES)} / {'-'.join(str(k[a]) for a in AXES)}")
    print("\nRough reading: same side at 80% or more means the judge can be trusted on that scale.")


if __name__ == "__main__":
    {"make": make, "compare": compare}[sys.argv[1] if len(sys.argv) > 1 else "make"]()
