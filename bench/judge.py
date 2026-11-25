"""A judge model that reads an answer and scores its style and its content.

It replaces the word lists in personas.py where those are too crude: it rates each of
the four live axes from 1 to 5 and checks the answer against the question's fact sheet,
including advice the answer added that the fact sheet does not contain.

    python -m bench.judge check     # how often the judge agrees with the style an answer was asked for

The key is read from GEMINI_API_KEY, or from the file .judge_key.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, ThinkingConfig
from pydantic import BaseModel, Field

MODEL = "gemini-3.8-flash"
FALLBACKS = ("gemini-3.6-flash", "gemini-3.5-flash")  # tried in order when MODEL fails
AXES = ["warmth", "directiveness", "formality", "depth"]  # a rating of 5 is the axis's first option

RUBRIC = """You are grading a short answer that an advisor gave to a person's question. The advisor was \
given a fact sheet and told to base the advice only on it.

Rate the answer's style on four scales from 1 to 5. Use 3 when the answer is mixed or shows neither end.
- warmth: 5 = opens with a remark that recognises how the person may be feeling; 1 = goes straight to the \
advice with no greeting, sympathy or encouragement.
- directiveness: 5 = gives the advice as commands (sentences that start with a verb, "do this"); 1 = describes \
what works for most people in statements and leaves the decision to the person.
- formality: 5 = relaxed and conversational, with contractions and everyday phrasing; 1 = formal, with no \
contractions.
- depth: 5 = gives a reason for each piece of advice; 1 = states the advice only, with no reasons.

Then check the content.
- facts_covered: for each point on the fact sheet, in order, whether the answer gives the advice that \
point contains. Paraphrase counts, and the point's supporting reason does not have to appear.
- added: every piece of advice or factual claim in the answer that is not on the fact sheet. Do not list \
remarks that only set the tone, or reasons that merely explain a point on the fact sheet. Empty if none."""


class Verdict(BaseModel):
    warmth: int = Field(ge=1, le=5)
    directiveness: int = Field(ge=1, le=5)
    formality: int = Field(ge=1, le=5)
    depth: int = Field(ge=1, le=5)
    facts_covered: list[bool]
    added: list[str]


def client():
    key = os.environ.get("GEMINI_API_KEY") or Path(".judge_key").read_text().strip()
    return genai.Client(api_key=key, http_options=HttpOptions(timeout=30_000))  # ms; a hung call must not stall a round


def judge(c, question, facts, answer, model=MODEL):
    """One answer's Verdict, as a dict."""
    prompt = (f"Question: {question}\n\nFact sheet:\n" + "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
              + f"\n\nAnswer:\n{answer}")
    # Low thinking: half the time and a third of the thinking tokens of the default, with the
    # same verdicts on the answers it was compared on. 3.8 Flash cannot switch thinking off.
    config = GenerateContentConfig(system_instruction=RUBRIC, temperature=0.0,
                                   response_mime_type="application/json", response_schema=Verdict,
                                   thinking_config=ThinkingConfig(thinking_level="low"))
    # Try each model in turn; a model that is overloaded, rate-limited, unreachable or
    # retired, or that returns something unreadable, gets one more try before the next.
    last = None
    for m in dict.fromkeys([model, *FALLBACKS]):
        for attempt in range(2):
            try:
                r = c.models.generate_content(model=m, contents=prompt, config=config)
                v = Verdict.model_validate_json(r.text).model_dump()
                if len(v["facts_covered"]) != len(facts):
                    raise ValueError("wrong number of fact checks")
                return {**v, "model": m}
            except Exception as e:  # noqa: BLE001 - any failure moves on; the caller has its own fallback
                last = e
                time.sleep(1 + attempt)
    raise RuntimeError(f"judge unavailable on every model: {type(last).__name__}: {str(last)[:200]}")


def judge_many(items, model=MODEL, workers=8):
    """items: (question, facts, answer) tuples. Verdicts in the same order."""
    c = client()
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda it: judge(c, *it, model=model), items))


def style_score(verdict, wants):
    """How well an answer suits a user. wants: axis -> +1 (likes the first option) or -1."""
    return sum(sign * (verdict[axis] - 3) for axis, sign in wants.items())


def check(report="results/modal/menu_check13/collab_statements_7B.json", scenario="data/advice.json"):
    """Agreement between the judge and the style each sample answer was asked to use.

    The samples come from the menu check, where the untouched model was given a known
    style. It does not always follow it, so this is a floor on the judge's accuracy.
    """
    from .personas import features
    sc = json.loads(Path(scenario).read_text())
    facts = {it["q"]: it["facts"] for split in ("train", "heldout", "test") for it in sc[split]}
    menu = {m["axis"]: m for m in sc["menu"] if m["axis"] in AXES}
    samples = json.loads(Path(report).read_text())["samples"]
    verdicts = judge_many([(s["q"], facts[s["q"]], s["text"]) for s in samples])
    agree = {a: [0, 0] for a in AXES}   # judge
    words = {a: [0, 0] for a in AXES}   # word lists, split at the midpoint of the two options' averages
    mid = {"warm_words": 0.22, "directive": 0.67, "casual": 0.195, "reasons": 0.27}
    for s, v in zip(samples, verdicts):
        f = features(s["text"])
        for a in AXES:
            first = menu[a]["options"][0][1] in s["style"]
            if v[a] != 3:
                agree[a][0] += (v[a] > 3) == first
                agree[a][1] += 1
            feat = menu[a]["feature"]
            words[a][0] += (f[feat] > mid[feat]) == first
            words[a][1] += 1
    for a in AXES:
        print(f"{a:14} judge agrees {agree[a][0]}/{agree[a][1]} (rated 3 on {len(samples) - agree[a][1]}) | "
              f"word lists {words[a][0]}/{words[a][1]}")
    n = len(samples)
    print(f"facts covered {sum(sum(v['facts_covered']) / len(v['facts_covered']) for v in verdicts) / n:.2f} | "
          f"answers with additions {sum(bool(v['added']) for v in verdicts)}/{n}")
    return samples, verdicts


def content(grid, prefix="", scenario="data/advice.json"):
    """Judge the content of what a grid's sessions showed and produced, by round.

    Shown: the four answers of every round. Tuned: the model's answers to the test
    questions before round one, after round five and at the end.
    """
    import glob
    from collections import defaultdict
    sc = json.loads(Path(scenario).read_text())
    cache_file = Path(f"results/judge/{grid}.json")
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    rows = []   # (run, kind, round, question, facts, text)
    for f in sorted(glob.glob(f"results/modal/{grid}/{prefix}*.json")):
        d = json.loads(Path(f).read_text())
        if "rounds" not in d:
            continue
        test = sc[d["args"]["eval_split"]]
        for r in d["rounds"]:
            item = sc["train"][(r["round"] - 1) % len(sc["train"])]
            rows += [(d["name"], "shown", r["round"], item["q"], item["facts"], t) for t in r["candidates"]]
        for rnd, texts in [(0, d["init"]["texts"]), (5, d["rounds"][4]["texts"]), (10, d["rounds"][-1]["texts"])]:
            rows += [(d["name"], "tuned", rnd, it["q"], it["facts"], t) for it, t in zip(test, texts)]
    todo = [r for r in rows if r[3] + "|" + r[5] not in cache]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), 40):   # save as we go, so an outage part-way loses nothing
        chunk = todo[start:start + 40]
        for r, v in zip(chunk, judge_many([r[3:] for r in chunk])):
            cache[r[3] + "|" + r[5]] = v
        cache_file.write_text(json.dumps(cache))
    agg = defaultdict(lambda: [0, 0.0, 0, 0])   # n, facts covered, answers missing a fact, answers with additions
    for run, kind, rnd, q, facts, text in rows:
        v = cache[q + "|" + text]
        bucket = (kind, "1-5" if rnd <= 5 else "6-10") if kind == "shown" else (kind, rnd)
        a = agg[bucket]
        a[0] += 1
        a[1] += sum(v["facts_covered"]) / len(facts)
        a[2] += not all(v["facts_covered"])
        a[3] += bool(v["added"])
    print(f"{grid} {prefix}: {len(rows)} answers judged ({len(todo)} new)")
    for (kind, rnd), (n, cov, miss, add) in sorted(agg.items(), key=str):
        print(f"  {kind:5} round {str(rnd):5} n {n:4} | facts covered {cov / n:.2f} | missing a fact {miss / n:.0%} | "
              f"adds something {add / n:.0%}")


def compare(grid, scenario="data/advice.json"):
    """Score each run's final answers to the test questions, grouped by the run name's prefix.

    Works for any method, because it reads the answers and not the model's internals:
    how many axes the answers sit on the side the simulated user wanted, how much better
    they suit that user than the untouched model's, and what happened to the content.
    """
    import glob
    from collections import defaultdict

    from .personas import PERSONAS
    sc = json.loads(Path(scenario).read_text())
    feature = {m["axis"]: m["feature"] for m in sc["menu"]}
    cache_file = Path(f"results/judge/{grid}_compare.json")
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"results/modal/{grid}/*.json"))
            if not f.endswith("calls.json")]
    runs = [d for d in runs if "rounds" in d]
    need = {}
    for d in runs:
        for texts in (d["base"]["texts"], d["rounds"][-1]["texts"]):
            for it, t in zip(sc[d["args"]["eval_split"]], texts):
                need.setdefault(it["q"] + "|" + t, (it["q"], it["facts"], t))
    todo = [k for k in need if k not in cache]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), 40):
        chunk = todo[start:start + 40]
        for k, v in zip(chunk, judge_many([need[k] for k in chunk])):
            cache[k] = v
        cache_file.write_text(json.dumps(cache))
    agg = defaultdict(lambda: defaultdict(list))
    for d in runs:
        signs = PERSONAS[d["args"]["persona"]].signs
        wants = {a: signs[feature[a]] for a in AXES}
        items = sc[d["args"]["eval_split"]]

        def summary(texts):
            vs = [cache[it["q"] + "|" + t] for it, t in zip(items, texts)]
            n = len(vs)
            return {"axes": sum((sum(v[a] for v in vs) / n - 3) * w > 0 for a, w in wants.items()),
                    "suit": sum(style_score(v, wants) for v in vs) / n,
                    "facts": sum(sum(v["facts_covered"]) / len(v["facts_covered"]) for v in vs) / n,
                    "adds": sum(bool(v["added"]) for v in vs) / n}
        tag = d["name"].rsplit("_", 3)[0]
        for name, texts in (("untouched model", d["base"]["texts"]), (tag, d["rounds"][-1]["texts"])):
            for k, v in summary(texts).items():
                agg[name][k].append(v)
    print(f"{grid}: {len(runs)} runs, {len(need)} answers judged ({len(todo)} new)")
    print(f"  {'':18} runs | axes on the wanted side (of 4) | suits the user (-8 to 8) | facts covered | adds something")
    def sd(v):
        mu = sum(v) / len(v)
        return (sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
    for name, a in agg.items():
        m = {k: sum(v) / len(v) for k, v in a.items()}
        print(f"  {name:18} {len(a['axes']):4} | {m['axes']:21.2f} (sd {sd(a['axes']):.2f}) | "
              f"{m['suit']:15.2f} (sd {sd(a['suit']):.2f}) | {m['facts']:13.2f} | {m['adds']:.0%}")
    fell_back = sum(not r["judged"] for d in runs for r in d["rounds"])
    print(f"  rounds where the judge was unavailable and word lists picked instead: {fell_back}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["compare"]:
        compare(*sys.argv[2:])
    if sys.argv[1:2] == ["content"]:
        content(*sys.argv[2:])

    if sys.argv[1:] == ["check"]:
        samples, verdicts = check()
        Path("results").mkdir(exist_ok=True)
        Path("results/judge_check.json").write_text(json.dumps(
            [{**s, "verdict": v} for s, v in zip(samples, verdicts)], indent=1))
