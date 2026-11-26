"""A live Attune session on Modal: a real person gives the feedback.

One GPU container holds the model and every open session's adapter. A plain test
page is served from the same app; the real interface will live elsewhere.

    modal deploy modal_live.py        # prints the URL; open it with ?key=<contents of .live_key>

Each round the page shows four short answers and the person marks the best and the
worst. To hide the wait, the server works one round ahead: while the person reads
round N, it applies their round N-1 feedback and then writes round N+1's answers.
So a round's feedback first shapes the answers two rounds later. After every update
the server records what the model now prefers and how it answers one fixed question,
and the page charts that at the end.
"""
import secrets as pysecrets
from pathlib import Path

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
AXES = ["warmth", "directiveness", "formality", "depth"]
ROUNDS, K = 5, 4
MEASURE, EXAMPLES = 6, 3  # unseen questions rated after every round; how many are shown as before/after
SHOWN = dict(temperature=0.7, top_p=0.95)   # how the answers on the results screen are written
LONG_ROUNDS = 10  # optional longer session: each of the ten questions once

# Access key: generated once, kept out of git, sent to the container as a secret.
_key_file = Path(__file__).parent / ".live_key"
if modal.is_local() and not _key_file.exists():
    _key_file.write_text(pysecrets.token_urlsafe(16))
KEY = _key_file.read_text().strip() if _key_file.exists() else ""

app = modal.App("attune-live")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.6", "transformers>=4.51", "peft>=0.15", "accelerate", "fastapi[standard]",
                 "google-genai>=1.56.0")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("bench")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_file("live_page.html", remote_path="/root/live_page.html")
)
hf_cache = modal.Volume.from_name("attune-hf-cache", create_if_missing=True)
# Finished sessions' adapters, packaged for download. Kept for a day.
adapters = modal.Volume.from_name("attune-adapters", create_if_missing=True)
KEEP_ADAPTERS_SECONDS = 24 * 3600

PACKAGE_README = """# Your Attune model

This is a LoRA adapter for {base}, tuned on {date} in a {rounds}-round Attune session.

- `adapter_model.safetensors`, `adapter_config.json`: the adapter, in the standard PEFT format.
- `attune.json`: your profile, the style the tuned model chose, and the prompt format it was tuned with.
- `example.py`: loads the base model with your adapter and rewrites or answers a piece of text in your style.

What was tuned: the model learned which of sixteen style combinations you preferred, and its
wording within that style, from the answers you marked best and least. It was tuned on short
pieces of advice written from a fact sheet, so that is where it will behave most like your session.
The answers are a demonstration of style, not advice.

Requirements: `pip install torch transformers peft`, and a GPU or Apple-silicon Mac with about 16 GB free.
"""

PACKAGE_EXAMPLE = """\"\"\"Answer a question in your Attune style: python example.py "How do I ask for a raise?" \"\"\"
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

here = Path(__file__).parent
meta = json.loads((here / "attune.json").read_text())
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(meta["base_model"])
base = AutoModelForCausalLM.from_pretrained(meta["base_model"], dtype=torch.bfloat16 if device == "cuda" else torch.float32)
model = PeftModel.from_pretrained(base, str(here)).to(device).eval()

question = sys.argv[1] if len(sys.argv) > 1 else "How do I ask for a raise?"
style = "\\n".join(f"- {c}" for c in meta["style_instructions"])
messages = [{"role": "system", "content": meta["system_prompt"]},
            {"role": "user", "content": f"Question: {question}\\n\\nWrite your answer in this style:\\n{style}"}]
# return_dict=True works across transformers versions (newer ones return a mapping, not a bare tensor)
enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
# a little randomness, so answers do not all open with the same phrase
out = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.95)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
"""


def key_ok(given, expected):
    import hmac
    return bool(expected) and hmac.compare_digest(given.encode(), expected.encode())


@app.cls(image=image, gpu="L40S", volumes={"/cache": hf_cache, "/adapters": adapters}, max_containers=1,
         scaledown_window=300,
         # the judge model's key, for rating the tuned model's writing after each round
         timeout=900, secrets=[modal.Secret.from_dict({"ATTUNE_KEY": KEY}), modal.Secret.from_name("attune-judge")])
@modal.concurrent(max_inputs=8)
class Live:
    @modal.enter()
    def load(self):
        import argparse
        import json
        import os
        import threading

        os.chdir("/root")
        from bench import run as R
        from bench.policy import Policy

        self.R = R
        self.lock = threading.Lock()  # one GPU, one piece of work at a time
        builtin = json.loads(Path("data/advice.json").read_text())
        self.menu = [m for name in AXES for m in builtin["menu"] if m["axis"] == name]
        R.use_menu(builtin["menu"], AXES)
        # The built-in questions: used when a visitor gives no setting, or the setting cannot be prepared.
        # "heldout" questions are never used in rounds; they are what the model is rated on.
        self.builtin = {"system": builtin["system"], "train": builtin["train"], "heldout": builtin["test"][:MEASURE]}
        self.args = argparse.Namespace(two_step=True, k=K, max_new_tokens=128, max_sentences=4,
                                       min_facts=1.0, strict_facts=True, redraws=3, neutral_start=True,
                                       # the four answers come from styles that differ on two scales or
                                       # more, and a near-copy of another answer is replaced
                                       min_distance=2, no_duplicates=True)
        self.policy = Policy(MODEL)
        self.fresh = self.policy.snapshot()  # the untouched adapter every session starts from
        self.sessions = {}
        hf_cache.commit()

    def _open(self, s):
        """Set a new session up on its own questions: prompts, and the standard model's answers to
        the rating questions (written here, rated in the background) as the "before"."""
        import threading
        s["sess"] = self.R.Session(self.policy, s["scenario"], self.args, latent=True)
        # Sampled, not the single most likely wording: the same setting as the tuned model's answers below.
        items = s["scenario"]["heldout"]
        outs = self.policy.generate(s["sess"].plain["heldout"], max_new_tokens=self.args.max_new_tokens,
                                    use_adapter=False, **SHOWN)
        s["base_texts"] = s["sess"].finish(outs, items=items)[1]
        s["written0"] = None
        s["rating0"] = threading.Thread(
            target=lambda: s.update(written0=self._written(s["scenario"]["heldout"], s["base_texts"])))
        s["rating0"].start()
        # Before any feedback every style is equally likely, so there is no lean to measure yet.
        s["history"] = [{"round": 0, "lean": {a: 0.0 for a in AXES}, "style": [], "clauses": [],
                         "answer": s["base_texts"][0], "texts": None}]

    # ---- pieces of GPU work; each runs with one session's adapter loaded -------------

    def _with(self, s, fn):
        with self.lock:
            self.policy.restore(s["weights"])
            try:
                return fn()
            finally:
                s["weights"] = self.policy.snapshot()

    def _written(self, items, texts):
        """The judge's average rating of these answers to the unseen questions, per axis (1 to 5).

        None if the judge is unavailable; the page then falls back to the model's style choices.
        """
        try:
            from bench.judge import judge_many
            verdicts = judge_many([(it["q"], it["facts"], t) for it, t in zip(items, texts)])
            return {a: round(sum(v[a] for v in verdicts) / len(verdicts), 2) for a in AXES}
        except Exception as e:  # noqa: BLE001 - a judge outage must not end the session
            print(f"judge unavailable: {e}", flush=True)
            return None

    def _track(self, s):
        """What the session's adapter prefers on each axis, and its answers to the rating questions."""
        import torch

        from bench.personas import WARM_PHRASES, features

        R, n_styles = self.R, len(self.R.STYLES)
        # The style choice is read on three of the unseen questions (it is the slow part, and it is
        # no longer what the page plots); the style favoured across them is used for every answer.
        sess = s["sess"]
        prompts = sess.styled["heldout"][:3]
        lean, total = {a: 0.0 for a in AXES}, 0
        for j, p in enumerate(prompts):
            lp = sess.style_logp(p)
            total = total + lp
            for a in AXES:
                m = torch.tensor(R.AXES[a])
                # log-odds of the axis's first option over its second
                lean[a] += (torch.logsumexp(lp[m], 0) - torch.logsumexp(lp[~m], 0)).item() / len(prompts)
        items = s["scenario"]["heldout"]
        # The answers that get rated are written in styles shared out in proportion to the model's
        # preferences (largest remainders, so nothing is random): a model split between two styles
        # writes some answers in each, and its rating sits in the middle instead of flipping.
        share = torch.softmax(total / len(prompts), dim=0) * len(items)
        counts = share.floor()
        for j in (share - counts).argsort(descending=True)[: int(len(items) - counts.sum())]:
            counts[j] += 1
        firsts = [j for j in share.argsort(descending=True).tolist() for _ in range(int(counts[j]))]
        # Sampled with a little randomness. Taking the single most likely wording every time makes
        # every answer open with the same phrase ("I get you're…"), whichever question it is.
        outs = self.policy.generate([sess.answer_prompt(it, f) for it, f in zip(items, firsts)],
                                    max_new_tokens=self.args.max_new_tokens, **SHOWN)
        texts = sess.finish(outs, firsts, items)[1]
        text, first = texts[0], firsts[0]
        style = [m["options"][0 if R.AXES[m["axis"]][first] else 1][0] for m in self.menu]
        f = features(text)
        return {"lean": {a: round(v, 2) for a, v in lean.items()}, "style": style, "answer": text, "texts": texts,
                "clauses": list(R.STYLE_CLAUSES[first]),
                "words": f["words"], "sentences": f["sentences"], "exclamations": text.count("!"),
                "warm_phrases": sum(text.lower().count(p) for p in WARM_PHRASES)}

    def _write(self, s, n):
        """Write round n's candidates with the session's current adapter."""
        import time
        t = time.time()
        i = (n - 1) % len(s["scenario"]["train"])
        picks, answers, texts, redrawn, segments = s["sess"].propose(i, s["rng"])
        s["cands"][n] = {"answers": answers, "segments": segments,
                         "payload": {"session": s["id"], "round": n, "rounds": s["rounds"],
                                     "question": s["scenario"]["train"][i]["q"], "candidates": texts,
                                     "role": s["role"], "builtin": s["builtin"], "notice": s["notice"],
                                     "write_seconds": round(time.time() - t, 1)}}

    def _learn(self, s, n, best, worst):
        """Apply round n's feedback, then record what the model now prefers."""
        import time

        from bench.methods import Group
        t = time.time()
        c = s["cands"].pop(n)
        rewards = [0.0] * K
        rewards[best], rewards[worst] = 1.0, -1.0
        s["method"].update([Group(None, c["answers"], list(range(K)), None, None,
                                  segments=c["segments"], rewards=rewards)], None)
        # Nothing is measured between rounds: the model is rated once, at the end (see _summary).

    def _after(self, s, n, best, worst):
        """Background work after round n's feedback: learn from it, then write round n+2."""
        try:
            self._with(s, lambda: self._learn(s, n, best, worst))
            if n + 2 <= s["rounds"]:
                self._with(s, lambda: self._write(s, n + 2))
        except Exception as e:  # surfaced on the next request
            s["error"] = repr(e)[:300]

    def _summary(self, s):
        # Measured once, at the end: the tuned model answers the unseen questions and the judge rates
        # each scale. The standard model's ratings on the same questions (taken at start-up) are the
        # comparison. If the judge is unavailable the page falls back to the model's style choices.
        final = {"round": s["rounds"], **self._track(s)}
        h = [s["history"][0], final]
        s["rating0"].join(timeout=40)
        written = [s["written0"], self._written(s["scenario"]["heldout"], final["texts"])]
        rated = all(written)
        axes = [{"axis": m["axis"], "first": m["options"][0][0], "second": m["options"][1][0],
                 "lean": [r["lean"][m["axis"]] for r in h],
                 "written": [w[m["axis"]] for w in written] if rated else None} for m in self.menu]
        texts = h[-1]["texts"]   # the tuned model's answers to the unseen questions, as rated
        examples = [{"question": it["q"], "before": b, "after": t}
                    for it, b, t in list(zip(s["scenario"]["heldout"], s["base_texts"], texts))[:EXAMPLES]]
        keep = ("round", "style", "answer", "words", "sentences", "exclamations", "warm_phrases", "update_seconds")
        return {"done": True, "session": s["id"], "rounds": s["rounds"], "axes": axes, "role": s["role"],
                "builtin": s["builtin"],
                "style_instructions": h[-1]["clauses"], "tracked_question": s["scenario"]["heldout"][0]["q"],
                "track": [{k: r.get(k) for k in keep} for r in h], "examples": examples}

    def _export(self, s, summary):
        """Package the session's adapter as a zip on the adapters volume, for download."""
        import json
        import shutil
        import time
        import zipfile

        root = Path("/adapters")
        d = root / s["id"]
        d.mkdir(parents=True, exist_ok=True)
        self.policy.model.save_pretrained(str(d))   # PEFT writes the adapter only, not the base model
        meta = {"base_model": MODEL, "created": time.strftime("%Y-%m-%d"), "rounds": s["rounds"],
                "profile": [{"axis": a["axis"], "first": a["first"], "second": a["second"], "lean": a["lean"][-1]}
                            for a in summary["axes"]],
                "speaking_as": s["role"], "questions": [it["q"] for it in s["scenario"]["train"][: s["rounds"]]],
                "style_instructions": summary["style_instructions"], "system_prompt": s["scenario"]["system"],
                "prompt_format": "Question, then the fact sheet if any, then 'Write your answer in this style:' "
                                 "followed by the style instructions as a bulleted list, all in the user turn."}
        (d / "attune.json").write_text(json.dumps(meta, indent=1))
        (d / "README.md").write_text(PACKAGE_README.format(base=MODEL, date=meta["created"], rounds=s["rounds"]))
        (d / "example.py").write_text(PACKAGE_EXAMPLE)
        with zipfile.ZipFile(root / f"{s['id']}.zip.part", "w", zipfile.ZIP_STORED) as z:
            for f in sorted(d.iterdir()):
                z.write(f, f"attune-model/{f.name}")
        (root / f"{s['id']}.zip.part").rename(root / f"{s['id']}.zip")
        shutil.rmtree(d)
        for old in root.glob("*.zip"):   # downloads are kept for a day
            if time.time() - old.stat().st_mtime > KEEP_ADAPTERS_SECONDS:
                old.unlink()
        adapters.commit()

    # ---- web endpoints ---------------------------------------------------------------

    @modal.asgi_app()
    def web(self):
        import os
        import random
        import threading
        import time
        import uuid

        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel

        from bench.methods import GRPOStyle

        api = FastAPI()

        def check(key):
            if not key_ok(key, os.environ.get("ATTUNE_KEY", "")):
                raise HTTPException(status_code=403, detail="Wrong or missing key")

        @api.get("/", response_class=HTMLResponse)
        def page(key: str = ""):
            check(key)
            return Path("/root/live_page.html").read_text()

        @api.get("/api/warm")
        def warm(key: str = ""):
            """Does nothing; calling it starts the GPU container while the visitor is still choosing."""
            check(key)
            return {"ready": True}

        class DraftItem(BaseModel):
            question: str
            points: list[str] | None = None   # None when the visitor wrote or edited the question

        class DraftIn(BaseModel):
            role: str
            items: list[DraftItem]
            measure: list[DraftItem]

        class StartIn(BaseModel):
            rounds: int | None = None
            draft: DraftIn | None = None

        def scenario_from(draft, rounds):
            """The session's questions from a visitor's draft; fact sheets are written here for
            any question they typed themselves. Returns (scenario, role)."""
            from bench import setting
            role = setting.clean(draft.role, 80).rstrip(".") or "an advisor"
            if len(draft.items) != rounds or len(draft.measure) != MEASURE or any(i.points is None for i in draft.measure):
                raise HTTPException(status_code=400, detail="That set of questions is incomplete; start again")
            items = [{"question": setting.clean(i.question, 200),
                      "points": None if i.points is None or len(i.points) != 2 else [setting.clean(p, 200) for p in i.points]}
                     for i in [*draft.items, *draft.measure]]
            if any(len(i["question"]) < 8 for i in items):
                raise HTTPException(status_code=400, detail="One of the questions is too short")
            typed = [i for i in items if i["points"] is None]
            if typed:
                try:
                    written = setting.points_for(role, [i["question"] for i in typed])
                except Exception:  # noqa: BLE001
                    raise HTTPException(status_code=502, detail="Could not prepare your own questions just now. "
                                                                "Try again, or use the suggested ones.")
                if not written["suitable"]:
                    raise HTTPException(status_code=400, detail=written["reason"] or "One of the questions is not suitable")
                for i, w in zip(typed, written["items"]):
                    i["points"] = w["points"]
            made = [setting.scenario_item(i) for i in items]
            # The model plays the role and speaks to the visitor directly, not as a third party.
            opening = "You are an advisor answering a person's question."
            assert opening in self.builtin["system"]
            system = self.builtin["system"].replace(
                opening, f"You are {role}. The person has come to you with a question. Answer them directly, "
                         f"speaking as yourself in that role.", 1)
            return {"system": system, "train": made[:rounds], "heldout": made[rounds:]}, role

        @api.post("/api/start")
        def start(body: StartIn | None = None, key: str = "", rounds: int = ROUNDS):
            check(key)
            rounds = (body and body.rounds) or rounds
            if rounds not in (ROUNDS, LONG_ROUNDS):
                raise HTTPException(status_code=400, detail=f"Rounds must be {ROUNDS} or {LONG_ROUNDS}")
            scenario, role, builtin = self.builtin, "an advisor", True
            if body and body.draft:
                scenario, role = scenario_from(body.draft, rounds)
                builtin = False
            s = {"id": uuid.uuid4().hex, "round": 0, "rounds": rounds, "rng": random.Random(), "weights": self.fresh,
                 "scenario": scenario, "role": role, "builtin": builtin, "notice": None,
                 "cands": {}, "error": None, "thread": None,
                 # 2e-5 for the five-question session: in the benchmark it kept nearly all of what 3e-5
                 # learned (3.58 against 3.67 axes of 4) with the wording staying close to the standard
                 # model's (drift 0.36 against 0.62), where 3e-5 settled on stock phrases in live sessions.
                 # Ten rounds stay at 1e-5.
                 # The wording is trained at a third of the strength of the style choice: in the
                 # benchmark that cut how often the final answers shared one opening (29% against 39%).
                 "method": GRPOStyle(self.policy, lr=2e-5 if rounds == ROUNDS else 1e-5, steps_per_round=8,
                                     choice_clip=0.2, answer_weight=0.3)}
            self.sessions[s["id"]] = s
            self._with(s, lambda: self._open(s))
            self._with(s, lambda: self._write(s, 1))
            if rounds >= 2:  # get one round ahead straight away
                s["thread"] = threading.Thread(target=lambda: self._with(s, lambda: self._write(s, 2)))
                s["thread"].start()
            return s["cands"][1]["payload"]

        class Feedback(BaseModel):
            session: str
            best: int
            worst: int

        @api.post("/api/feedback")
        def feedback(fb: Feedback, key: str = ""):
            check(key)
            s = self.sessions.get(fb.session)
            if s is None:
                raise HTTPException(status_code=404, detail="Session expired; start again")
            if fb.best == fb.worst or not {fb.best, fb.worst} <= set(range(K)):
                raise HTTPException(status_code=400, detail="Pick two different answers")
            t = time.time()
            if s["thread"]:
                s["thread"].join()  # wait for the work started after the previous round
            if s["error"]:
                raise HTTPException(status_code=500, detail=s["error"])
            n = s["round"] = s["round"] + 1
            if n < s["rounds"]:
                payload = {**s["cands"][n + 1]["payload"], "waited_seconds": round(time.time() - t, 1)}
                s["thread"] = threading.Thread(target=self._after, args=(s, n, fb.best, fb.worst))
                s["thread"].start()
                return payload
            self._after(s, n, fb.best, fb.worst)  # last round: nothing left to hide the wait behind
            if s["error"]:
                raise HTTPException(status_code=500, detail=s["error"])
            out = self._with(s, lambda: self._summary(s))
            del self.sessions[s["id"]]
            # Package the adapter in the background; the download appears a few seconds later.
            threading.Thread(target=lambda: self._with(s, lambda: self._export(s, out))).start()
            return {**out, "waited_seconds": round(time.time() - t, 1)}

        return api


@app.function(image=image, volumes={"/adapters": adapters},
              secrets=[modal.Secret.from_dict({"ATTUNE_KEY": KEY}), modal.Secret.from_name("attune-judge")])
@modal.asgi_app()
def files():
    """The work that needs no GPU: preparing a visitor's setting, and serving packaged adapters."""
    import os
    import re

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    api = FastAPI()

    class PrepareIn(BaseModel):
        setting: str
        rounds: int = ROUNDS

    @api.post("/prepare")
    def prepare(body: PrepareIn, key: str = ""):
        """Check a visitor's setting and draft its questions: the session's, then the rating ones."""
        if not key_ok(key, os.environ.get("ATTUNE_KEY", "")):
            raise HTTPException(status_code=401, detail="Missing or wrong key")
        if body.rounds not in (ROUNDS, LONG_ROUNDS):
            raise HTTPException(status_code=400, detail=f"Rounds must be {ROUNDS} or {LONG_ROUNDS}")
        os.chdir("/root")
        from bench import setting
        if len(setting.clean(body.setting, 300)) < 3:
            raise HTTPException(status_code=400, detail="Say who the model should speak as")
        try:
            d = setting.draft(body.setting, body.rounds + MEASURE)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=502, detail="Could not prepare that setting just now")
        return {"suitable": d["suitable"], "reason": d["reason"], "role": d["role"],
                "items": d["items"][: body.rounds], "measure": d["items"][body.rounds:]}

    @api.get("/adapter/{session}")
    def adapter(session: str, key: str = ""):
        if not key_ok(key, os.environ.get("ATTUNE_KEY", "")):
            raise HTTPException(status_code=401, detail="Missing or wrong key")
        if not re.fullmatch(r"[0-9a-f]{32}", session):
            raise HTTPException(status_code=404, detail="No such session")
        adapters.reload()
        path = Path("/adapters") / f"{session}.zip"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Not ready yet, or older than a day")
        return FileResponse(path, media_type="application/zip", filename="attune-model.zip")

    return api

