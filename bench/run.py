"""Simulate one Attune session with a synthetic user and measure preference recovery.

    python -m bench.run --method dpo --persona warm_brief

Each round: the policy writes K candidates in different styles, the persona ranks
them, the method updates the adapter. After every round the tuned policy answers
held-out prompts and is compared to the base model.

Two ways of getting varied candidates:
  dpo, grpo   : a hidden style instruction is appended to the user turn (off-policy).
  grpo_style  : the policy opens its own reply with a style line chosen from a fixed
                menu, then answers in that style. The choice is part of the policy,
                so the candidates are on-policy and the choice itself is learnable.
"""
import argparse
import copy
import json
import random
import re
import time
from pathlib import Path

import torch

from .methods import METHODS, Group
from .personas import LIVE_RANGE, PERSONAS, by_live_axes, facts_retained, features, rank
from .policy import Policy

# Style menu. Worded concretely, because a small model ignores abstract instructions.
LENGTH = ["Answer in one or two short sentences.",
          "Write four or five sentences.",
          "Write eight to ten sentences with plenty of detail."]
TONE = ["Sound warm and encouraging: open with a kind remark about the question and talk to the student as 'you'.",
        "Sound neutral and formal: no greeting, no exclamation marks, and never address the student directly."]
DEVICE = ["End with a short check-in question for the student.",
          "Do not ask any questions."]
STYLES = [f"{a} {b} {c}" for a in LENGTH for b in TONE for c in DEVICE]
AXES = {"short": [s.startswith(LENGTH[0]) for s in STYLES],
        "long": [s.startswith(LENGTH[2]) for s in STYLES],
        "warm": [TONE[0] in s for s in STYLES],
        "question": [s.endswith(DEVICE[0]) for s in STYLES]}

# Optional fourth length, for users who want more than the standard menu's longest style.
LONGEST = "Write a very long, exhaustive answer of at least 200 words that elaborates on every fact."


def widen_menu():
    """Add LONGEST to the menu (16 styles instead of 12)."""
    global STYLES, AXES
    STYLES = [f"{a} {b} {c}" for a in LENGTH + [LONGEST] for b in TONE for c in DEVICE]
    AXES = {"short": [s.startswith(LENGTH[0]) for s in STYLES],
            "long": [s.startswith((LENGTH[2], LONGEST)) for s in STYLES],
            "warm": [TONE[0] in s for s in STYLES],
            "question": [s.endswith(DEVICE[0]) for s in STYLES]}


STYLE_CLAUSES = STYLE_CAPS = None  # per style: its instructions one by one, and its sentence cap


def use_menu(menu, axes):
    """Build the style menu from the named axes of a scenario's own menu definition."""
    global STYLES, AXES, STYLE_CLAUSES, STYLE_CAPS
    chosen = [m for name in axes for m in menu if m["axis"] == name]
    assert len(chosen) == len(axes), f"unknown axis in {axes}"
    combos = [[]]  # each style is one (axis, option index, instruction, sentence cap) per axis
    for m in chosen:
        combos = [c + [(m["axis"], i, o[1], o[2] if len(o) > 2 else 3)]
                  for c in combos for i, o in enumerate(m["options"])]
    STYLES = [" ".join(t for _, _, t, _ in c) for c in combos]
    STYLE_CLAUSES = [[t for _, _, t, _ in c] for c in combos]
    # An option may set its own sentence cap (e.g. "one sentence"); otherwise three.
    STYLE_CAPS = [min(cap for *_, cap in c) for c in combos]
    # p_<axis> is the probability of that axis's first option
    AXES = {m["axis"]: [c[j][1] == 0 for c in combos] for j, m in enumerate(chosen)}


def cap_sentences(text, k):
    """The first k sentences of text, with its line breaks kept."""
    text = text.strip()
    ends = [m.start() for m in re.finditer(r"(?<=[.!?])\s+", text)]
    return text if len(ends) < k else text[:ends[k - 1]]


STYLE_FORMAT = ("\nBegin your reply with one line that starts with 'Style:' and states the style "
                "you will use, then a blank line, then your answer written in exactly that style.")
EXPLORE = 0.3  # share of candidate styles drawn uniformly instead of from the policy


def user_message(item):
    return f"Question: {item['q']}\n\nFacts:\n" + "\n".join(f"- {f}" for f in item["facts"])


class Session:
    """Everything that differs between hidden-instruction and style-line candidates."""

    def __init__(self, policy, scenario, args, latent):
        self.policy, self.scenario, self.args, self.latent = policy, scenario, args, latent
        self.system = scenario["system"]
        splits = [s for s in ("train", "heldout", "probe") if s in scenario]
        self.plain = {s: [policy.prompt_ids(self.system, user_message(it)) for it in scenario[s]]
                      for s in splits}
        # Two-step policy: choose a style line, then answer a prompt that carries that
        # style as bullets in the user turn (where the model follows instructions best).
        self.two_step = latent and args.two_step
        self._base_lp = {}  # the untouched model's style scores per prompt, for --neutral-start
        if latent:
            self.styled = {s: [policy.prompt_ids(self.system + STYLE_FORMAT, user_message(it))
                               for it in scenario[s]] for s in splits}
            self.prefixes = [policy.tok(f"Style: {s}\n\n", add_special_tokens=False).input_ids for s in STYLES]

    def answer_prompt(self, item, s):
        bullets = "\n".join(f"- {c}" for c in STYLE_CLAUSES[s])
        return self.policy.prompt_ids(
            self.system, f"{user_message(item)}\n\nWrite your answer in this style:\n{bullets}")

    def finish(self, outs, styles=None, items=None):
        """Decode answers and enforce the sentence cap. Returns (token ids, texts).

        A style's own cap (one sentence) is relaxed, up to the overall limit, when
        cutting there would lose a fact the answer goes on to state.
        """
        texts = [self.policy.decode(o) for o in outs]
        limit = self.args.max_sentences
        if not limit:
            return outs, texts
        caps = [min(STYLE_CAPS[s], limit) if styles and STYLE_CAPS else limit for s in (styles or outs)]
        if items:
            for j, (t, it) in enumerate(zip(texts, items)):
                full = facts_retained(cap_sentences(t, limit), it["keys"])
                while caps[j] < limit and facts_retained(cap_sentences(t, caps[j]), it["keys"]) < full:
                    caps[j] += 1
        texts = [cap_sentences(t, c) for t, c in zip(texts, caps)]
        ids = [self.policy.tok(t, add_special_tokens=False).input_ids + [self.policy.eos_id] for t in texts]
        return ids, texts

    def train_prompt(self, i):
        return (self.styled if self.latent else self.plain)["train"][i]

    def style_logp(self, prompt):
        """Log-probability of the policy choosing each style line in the menu.

        With --neutral-start the choice is measured relative to the untouched model, so
        every style starts equally likely and only what was learned moves it.
        """
        lp = self.policy.logprobs([prompt] * len(STYLES), self.prefixes).float().cpu()
        if getattr(self.args, "neutral_start", False):
            key = tuple(prompt)
            if key not in self._base_lp:
                self._base_lp[key] = self.policy.logprobs([prompt] * len(STYLES), self.prefixes,
                                                          use_adapter=False).float().cpu()
            lp = lp - self._base_lp[key]
        return torch.log_softmax(lp, dim=0)

    def style_dist(self, prompt):
        return self.style_logp(prompt).exp()

    def propose(self, i, rng):
        """K candidates for training prompt i: (styles, token ids, texts, redraw count, segments)."""
        a, item = self.args, self.scenario["train"][i]
        if self.latent:
            dist = (1 - EXPLORE) * self.style_dist(self.styled["train"][i]) + EXPLORE / len(STYLES)
            g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
            picks = torch.multinomial(dist, a.k, replacement=False, generator=g).tolist()
        else:
            picks = rng.sample(range(len(STYLES)), a.k)

        def context(s):
            if self.two_step or (not self.latent and STYLE_CLAUSES):
                # The style as bullets in the user turn. For dpo this instruction is hidden:
                # the model is trained on the plain prompt to prefer the answers it led to.
                return self.answer_prompt(item, s)
            if self.latent:
                return self.styled["train"][i] + self.prefixes[s]
            return self.policy.prompt_ids(self.system, f"{user_message(item)}\n\nStyle: {STYLES[s]}")

        def draw(idx, use_adapter=True):
            outs = self.policy.generate([context(picks[j]) for j in idx], max_new_tokens=a.max_new_tokens,
                                        temperature=0.8, top_p=0.95, use_adapter=use_adapter)
            return self.finish(outs, [picks[j] for j in idx], [item] * len(idx))

        def failing():
            return [j for j, t in enumerate(texts) if facts_retained(t, item["keys"]) < a.min_facts]

        answers, texts = draw(range(a.k))
        # Only show candidates that still carry the fact sheet, so feedback can express
        # a style preference but never reward dropping content.
        redrawn = 0
        for attempt in range(a.redraws):
            bad = failing()
            if not bad:
                break
            redrawn += len(bad)
            if a.strict_facts and attempt == a.redraws - 1:
                # Still failing: this style cannot carry the facts here, so swap in another.
                for j in bad:
                    picks[j] = rng.choice([s for s in range(len(STYLES)) if s not in picks])
            # The last attempt falls back to the untouched base model.
            fresh, fresh_texts = draw(bad, use_adapter=attempt < a.redraws - 1)
            for j, o, t in zip(bad, fresh, fresh_texts):
                if a.strict_facts and attempt == a.redraws - 1 or \
                        facts_retained(t, item["keys"]) > facts_retained(texts[j], item["keys"]):
                    answers[j], texts[j] = o, t
        if self.two_step:
            # Each candidate is two pieces of the policy's output: the style it chose
            # and the answer it wrote under that style.
            choice = self.styled["train"][i]
            segments = [[(choice, self.prefixes[s]), (context(s), answers[j])] for j, s in enumerate(picks)]
            return picks, answers, texts, redrawn, segments
        cands = [self.prefixes[s] + o for s, o in zip(picks, answers)] if self.latent else answers
        return picks, cands, texts, redrawn, None

    def answer_heldout(self, use_adapter=True, plain=False, split="heldout"):
        """Greedy answers to the held-out prompts, plus the style marginals if latent."""
        if not self.latent or plain:
            outs = self.policy.generate(self.plain[split], max_new_tokens=self.args.max_new_tokens,
                                        use_adapter=use_adapter)
            return self.finish(outs, items=self.scenario[split])[1], {}
        dists = torch.stack([self.style_dist(p) for p in self.styled[split]])
        best = [d.argmax().item() for d in dists]
        if self.two_step:
            contexts = [self.answer_prompt(it, s) for it, s in zip(self.scenario[split], best)]
        else:
            contexts = [p + self.prefixes[s] for p, s in zip(self.styled[split], best)]
        outs = self.policy.generate(contexts, max_new_tokens=self.args.max_new_tokens)
        mean = dists.mean(0)
        marginals = {f"p_{k}": mean[torch.tensor(mask)].sum().item() for k, mask in AXES.items()}
        # How committed the choice is, and how far the wording has moved from the untouched
        # model's: the base model's average surprise per token at the tuned answers.
        marginals["style_top"] = dists.max(1).values.mean().item()
        marginals["style_entropy"] = -(dists * dists.clamp_min(1e-12).log()).sum(1).mean().item()
        base_lp = self.policy.logprobs(contexts, outs, use_adapter=False).float().cpu()
        marginals["drift"] = (-base_lp / torch.tensor([len(o) for o in outs])).mean().item()
        return self.finish(outs, best, self.scenario[split])[1], marginals


def probe_facts(sess, use_adapter=True):
    """Share of fact-sheet points kept on the probe questions: the guardrail's signal."""
    texts, _ = sess.answer_heldout(use_adapter=use_adapter, plain=not use_adapter, split="probe")
    items = sess.scenario["probe"]
    return sum(facts_retained(t, it["keys"]) for t, it in zip(texts, items)) / len(items)


def score_outputs(texts, scenario, persona, base_scores):
    scores = [PERSONAS[persona](t) for t in texts]
    result = {
        "score": sum(scores) / len(scores),
        "words": sum(features(t)["words"] for t in texts) / len(texts),
        "facts": sum(facts_retained(t, it["keys"]) for t, it in zip(texts, scenario["heldout"])) / len(texts),
        "texts": texts,
        "scores": scores,
    }
    if base_scores is not None:
        wins = [1.0 if s > b else 0.5 if s == b else 0.0 for s, b in zip(scores, base_scores)]
        result["win_rate"] = sum(wins) / len(wins)
    return result


def user_wants(persona, menu, flipped=False, flip_axes=None):
    """axis -> +1 or -1 for a persona with signs, after any reversal (None if it has no signs)."""
    signs = getattr(PERSONAS[persona], "signs", None)
    feature = {m["axis"]: m["feature"] for m in menu or []}
    if not signs or not all(feature.get(a) in signs for a in AXES):
        return None
    return {a: signs[feature[a]] * (-1 if flipped and (not flip_axes or a in flip_axes) else 1) for a in AXES}


def axes_right(marg, wants):
    """How many axes lean the way the simulated user wants."""
    return sum((marg[f"p_{a}"] > 0.5) == (wants[a] > 0) for a in AXES)


def line(r, ev, extra=""):
    win = f" | win {ev['win_rate']:.2f}" if "win_rate" in ev else ""
    return f"round {r:2d} | score {ev['score']:.2f}{win} | words {ev['words']:5.1f} | facts {ev['facts']:.2f}{extra}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--scenario", default="data/tutor.json")
    ap.add_argument("--method", choices=METHODS, default="dpo")
    ap.add_argument("--persona", choices=PERSONAS, default="warm_brief")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--menu", choices=["standard", "wide"], default="standard",
                    help="'wide' adds a fourth, longer length option to the style menu")
    ap.add_argument("--axes", default=None,
                    help="comma-separated axes from the scenario's menu, e.g. warmth,directiveness,brevity")
    ap.add_argument("--two-step", action="store_true",
                    help="grpo_style: choose a style, then answer with it placed in the user turn")
    ap.add_argument("--neutral-start", action="store_true",
                    help="grpo_style: start with every style equally likely instead of the model's defaults")
    ap.add_argument("--max-sentences", type=int, default=0,
                    help="cut every answer to this many sentences (0 = no cap)")
    ap.add_argument("--feedback", choices=["rank", "bestworst"], default="rank",
                    help="bestworst: the user only marks their favourite and least favourite candidate")
    ap.add_argument("--picks", type=int, default=0,
                    help="if >0 the user picks their favourite N candidates instead of ranking all K")
    ap.add_argument("--flip-at", type=int, default=0,
                    help="after this round the simulated user's preferences reverse (0 = never)")
    ap.add_argument("--flip-axes", default="",
                    help="comma-separated axes that reverse at --flip-at (default: all of them)")
    ap.add_argument("--judge-user", action="store_true",
                    help="the simulated user reads answers with the judge model instead of word lists; "
                         "falls back to the word lists for a round if the judge is unavailable")
    ap.add_argument("--choice-clip", type=float, default=0.0,
                    help="grpo_style: clip the style choice as one action, moving its probability by at most "
                         "this share per round (0 = clip token by token)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=8, help="gradient steps per round")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--min-facts", type=float, default=0.66,
                    help="share of fact-sheet points a candidate must keep to be shown (0 = no filter)")
    ap.add_argument("--strict-facts", action="store_true",
                    help="if a candidate still drops facts after redraws, replace its style with another")
    ap.add_argument("--redraws", type=int, default=3, help="attempts to replace a candidate that drops facts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-split", default="heldout",
                    help="'heldout' was used while developing the methods; 'test' is untouched")
    ap.add_argument("--heldout", type=int, default=None, help="limit held-out prompts (smoke tests)")
    ap.add_argument("--guard", type=float, default=0.0,
                    help="undo a round's update if fact retention on probe questions falls more than "
                         "this far below the base model's (0 = no guardrail)")
    ap.add_argument("--beta", type=float, default=None, help="DPO only: strength of the pull to the base model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.menu == "wide":
        widen_menu()
    if args.axes:
        use_menu(json.loads(Path(args.scenario).read_text())["menu"], args.axes.split(","))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    scenario = json.loads(Path(args.scenario).read_text())
    # Probe questions for the guardrail: never the ones used for evaluation or for rounds.
    scenario["probe"] = scenario["heldout"][:5] if args.eval_split != "heldout" else scenario["train"][10:12]
    scenario["heldout"] = scenario[args.eval_split]
    if args.heldout:
        scenario["heldout"] = scenario["heldout"][: args.heldout]

    t0 = time.time()
    policy = Policy(args.model)
    print(f"loaded {args.model} on {policy.device} in {time.time() - t0:.1f}s")
    extra = {"beta": args.beta} if args.beta is not None else {}
    if args.choice_clip:
        extra["choice_clip"] = args.choice_clip
    menu = scenario.get("menu") if args.axes else None
    flip_axes = set(filter(None, args.flip_axes.split(",")))
    method = METHODS[args.method](policy, lr=args.lr, steps_per_round=args.steps, seed=args.seed, **extra)
    latent = getattr(method, "latent", False)
    sess = Session(policy, scenario, args, latent)

    # The reference is always the plain base model with no style line.
    base = score_outputs(sess.answer_heldout(use_adapter=False, plain=True)[0], scenario, args.persona, None)
    print(line(0, base), flush=True)
    log = {"args": vars(args), "base": base, "rounds": []}
    if latent:
        texts, marg = sess.answer_heldout()
        init = {**score_outputs(texts, scenario, args.persona, base["scores"]), **marg}
        if user_wants(args.persona, menu):
            init["axes_right"] = axes_right(marg, user_wants(args.persona, menu))
        log["init"] = init
        print(line(0, init, " | untuned, with style line | " + " ".join(f"{k} {v:.2f}" for k, v in marg.items())),
              flush=True)

    probe_base = probe_facts(sess, use_adapter=False) if args.guard else None
    groups = []
    for r in range(1, args.rounds + 1):
        i = (r - 1) % len(scenario["train"])
        t_gen = time.time()
        picks, cands, cand_texts, redrawn, segments = sess.propose(i, rng)
        t_gen = time.time() - t_gen
        flipped = bool(args.flip_at) and r > args.flip_at
        wants = user_wants(args.persona, menu, flipped, flip_axes)
        judged = None
        if args.judge_user and wants:
            try:
                from .judge import judge_many, style_score
                item = scenario["train"][i]
                verdicts = judge_many([(item["q"], item["facts"], t) for t in cand_texts])
                marks = [style_score(v, wants) for v in verdicts]
                order = sorted(range(len(marks)), key=lambda j: -marks[j])
                judged = sorted({v["model"] for v in verdicts})
            except Exception as e:  # noqa: BLE001 - a judge outage must not end the session
                print(f"judge unavailable in round {r}, using word lists: {e}", flush=True)
        if judged is None:
            feature = {m["axis"]: m["feature"] for m in menu or []}
            if wants and all(feature[a] in LIVE_RANGE for a in wants):
                # The same user, read through the word lists (handles a partial reversal too).
                score = by_live_axes(**{feature[a]: w for a, w in wants.items()})
                order = sorted(range(len(cand_texts)), key=lambda j: -score(cand_texts[j]))
            else:
                order = rank(args.persona, cand_texts)
                if flipped:
                    order = order[::-1]

        t_upd = time.time()
        prompt = sess.train_prompt(i)
        if latent:
            rewards = None
            if args.feedback == "bestworst":
                rewards = [0.0] * args.k
                rewards[order[0]], rewards[order[-1]] = 1.0, -1.0
            groups.append(Group(prompt, cands, order, None, None, segments=segments, rewards=rewards,
                                liked=order[: args.picks] if args.picks else None))
        else:
            ref = policy.logprobs([prompt] * args.k, cands, use_adapter=False).detach().cpu()
            groups.append(Group(prompt, cands, order, ref, policy.style_features(cands),
                                liked=order[: args.picks] if args.picks else None,
                                bestworst=args.feedback == "bestworst"))
        if args.guard:
            saved = policy.snapshot(), copy.deepcopy(method.opt.state_dict())
        info = method.update(groups, sess.plain["train"])
        t_upd = time.time() - t_upd
        if args.guard:
            # Guardrail: if this update cost too much content, undo it and drop the round.
            t_guard = time.time()
            probe = probe_facts(sess)
            rolled_back = probe < probe_base - args.guard
            if rolled_back:
                policy.restore(saved[0])
                method.opt.load_state_dict(saved[1])
                groups.pop()
            info = {**info, "probe_facts": probe, "rolled_back": float(rolled_back),
                    "t_guard": time.time() - t_guard}

        texts, marg = sess.answer_heldout()
        ev = score_outputs(texts, scenario, args.persona, base["scores"])
        info = {**info, **marg}
        if latent and wants:
            # Judged against what the user wanted during this round.
            info["axes_right"] = axes_right(marg, wants)
        print(line(r, ev, f" | redrawn {redrawn} | gen {t_gen:4.1f}s | update {t_upd:5.1f}s | "
                   + " ".join(f"{k} {v:.3f}" for k, v in info.items())), flush=True)
        log["rounds"].append({"round": r, "styles": [STYLES[s] for s in picks], "candidates": cand_texts,
                              "redrawn": redrawn, "order": order, "flipped": flipped, "judged": judged, "t_generate": t_gen, "t_update": t_upd,
                              **info, **ev})

    out = Path(args.out or f"results/{args.method}_{args.persona}_s{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=1))
    print(f"saved {out}")

    last = log["rounds"][-1]
    for j in range(min(2, len(last["texts"]))):
        print(f"\nQ: {scenario['heldout'][j]['q']}\n  base : {base['texts'][j]}\n  tuned: {last['texts'][j]}")
    return log


if __name__ == "__main__":
    main()
