"""Aggregate benchmark runs across seeds.

    python -m bench.report results/bench
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results/bench")
    runs = defaultdict(list)
    for f in sorted(folder.glob("*.json")):
        d = json.loads(f.read_text())
        if "rounds" not in d:  # e.g. calls.json written by bench.remote
            continue
        runs[(d["args"]["persona"], d["args"]["method"])].append(d)

    def ms(xs):
        return f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"

    print(f"{'persona':16} {'method':11} n  {'base':>5}  {'final score':>12}  {'win rate':>12}  "
          f"{'facts':>12}  {'words':>6}  {'update s':>8}")
    for (persona, method), ds in sorted(runs.items()):
        last = [d["rounds"][-1] for d in ds]
        upd = [st.mean(r["t_update"] for r in d["rounds"]) for d in ds]
        print(f"{persona:16} {method:11} {len(ds)}  {ds[0]['base']['score']:5.2f}  "
              f"{ms([l['score'] for l in last]):>12}  {ms([l['win_rate'] for l in last]):>12}  "
              f"{ms([l['facts'] for l in last]):>12}  {st.mean(l['words'] for l in last):6.1f}  {st.mean(upd):8.1f}")
        print(f"{'':28} base facts {ds[0]['base']['facts']:.2f} | win rate by round: "
              + " ".join(f"{st.mean(d['rounds'][r]['win_rate'] for d in ds):.2f}"
                         for r in range(len(ds[0]['rounds']))))


if __name__ == "__main__":
    main()
