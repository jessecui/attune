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
LONG_ROUNDS = 10  # optional longer session: each of the ten questions once

# Access key: generated once, kept out of git, sent to the container as a secret.
_key_file = Path(__file__).parent / ".live_key"
if modal.is_local() and not _key_file.exists():
    _key_file.write_text(pysecrets.token_urlsafe(16))
KEY = _key_file.read_text().strip() if _key_file.exists() else ""

app = modal.App("attune-live")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.6", "transformers>=4.51", "peft>=0.15", "accelerate", "fastapi[standard]")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("bench")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_file("live_page.html", remote_path="/root/live_page.html")
)
hf_cache = modal.Volume.from_name("attune-hf-cache", create_if_missing=True)


def key_ok(given, expected):
    import hmac
    return bool(expected) and hmac.compare_digest(given.encode(), expected.encode())


@app.cls(image=image, gpu="L40S", volumes={"/cache": hf_cache}, max_containers=1, scaledown_window=300,
         timeout=900, secrets=[modal.Secret.from_dict({"ATTUNE_KEY": KEY})])
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
        self.scenario = json.loads(Path("data/advice.json").read_text())
        self.menu = [m for name in AXES for m in self.scenario["menu"] if m["axis"] == name]
        R.use_menu(self.scenario["menu"], AXES)
        # Questions for tracking and the before/after comparison; never used in rounds.
        self.scenario["heldout"] = self.scenario["test"][:3]
        self.args = argparse.Namespace(two_step=True, k=K, max_new_tokens=128, max_sentences=4,
                                       min_facts=1.0, strict_facts=True, redraws=3, neutral_start=True)
        self.policy = Policy(MODEL)
        self.fresh = self.policy.snapshot()  # the untouched adapter every session starts from
        self.sess = R.Session(self.policy, self.scenario, self.args, latent=True)
        self.base_texts = self.sess.answer_heldout(use_adapter=False, plain=True)[0]
        # Before any feedback every style is equally likely, so the starting answer shown
        # is the untouched model's, written with no style instructions.
        self.track0 = {"round": 0, **self._track(plain=self.base_texts[0])}
        self.sessions = {}
        hf_cache.commit()

    # ---- pieces of GPU work; each runs with one session's adapter loaded -------------

    def _with(self, s, fn):
        with self.lock:
            self.policy.restore(s["weights"])
            try:
                return fn()
            finally:
                s["weights"] = self.policy.snapshot()

    def _track(self, plain=None):
        """What the current adapter prefers on each axis, and its answer to one fixed question."""
        import torch

        from bench.personas import WARM_PHRASES, features

        R, n_styles = self.R, len(self.R.STYLES)
        prompts = self.sess.styled["heldout"]
        lean, first = {a: 0.0 for a in AXES}, None
        for j, p in enumerate(prompts):
            lp = self.sess.style_logp(p)
            if j == 0:
                first = int(lp.argmax())
            for a in AXES:
                m = torch.tensor(R.AXES[a])
                # log-odds of the axis's first option over its second
                lean[a] += (torch.logsumexp(lp[m], 0) - torch.logsumexp(lp[~m], 0)).item() / len(prompts)
        item = self.scenario["heldout"][0]
        if plain is None:
            out = self.policy.generate([self.sess.answer_prompt(item, first)],
                                       max_new_tokens=self.args.max_new_tokens)
            text = self.sess.finish(out, [first], [item])[1][0]
            style = [m["options"][0 if R.AXES[m["axis"]][first] else 1][0] for m in self.menu]
        else:
            text, style = plain, []
        f = features(text)
        return {"lean": {a: round(v, 2) for a, v in lean.items()}, "style": style, "answer": text,
                "words": f["words"], "sentences": f["sentences"], "exclamations": text.count("!"),
                "warm_phrases": sum(text.lower().count(p) for p in WARM_PHRASES)}

    def _write(self, s, n):
        """Write round n's candidates with the session's current adapter."""
        import time
        t = time.time()
        i = (n - 1) % len(self.scenario["train"])
        picks, answers, texts, redrawn, segments = self.sess.propose(i, s["rng"])
        s["cands"][n] = {"answers": answers, "segments": segments,
                         "payload": {"session": s["id"], "round": n, "rounds": s["rounds"],
                                     "question": self.scenario["train"][i]["q"], "candidates": texts,
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
        s["history"].append({"round": n, "update_seconds": round(time.time() - t, 1), **self._track()})

    def _after(self, s, n, best, worst):
        """Background work after round n's feedback: learn from it, then write round n+2."""
        try:
            self._with(s, lambda: self._learn(s, n, best, worst))
            if n + 2 <= s["rounds"]:
                self._with(s, lambda: self._write(s, n + 2))
        except Exception as e:  # surfaced on the next request
            s["error"] = repr(e)[:300]

    def _summary(self, s):
        texts, _ = self.sess.answer_heldout()
        h = s["history"]
        axes = [{"axis": m["axis"], "first": m["options"][0][0], "second": m["options"][1][0],
                 "lean": [r["lean"][m["axis"]] for r in h]} for m in self.menu]
        examples = [{"question": it["q"], "before": b, "after": t}
                    for it, b, t in zip(self.scenario["heldout"], self.base_texts, texts)]
        keep = ("round", "style", "answer", "words", "sentences", "exclamations", "warm_phrases", "update_seconds")
        return {"done": True, "axes": axes, "tracked_question": self.scenario["heldout"][0]["q"],
                "track": [{k: r.get(k) for k in keep} for r in h], "examples": examples}

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

        @api.post("/api/start")
        def start(key: str = "", rounds: int = ROUNDS):
            check(key)
            if rounds not in (ROUNDS, LONG_ROUNDS):
                raise HTTPException(status_code=400, detail=f"Rounds must be {ROUNDS} or {LONG_ROUNDS}")
            s = {"id": uuid.uuid4().hex, "round": 0, "rounds": rounds, "rng": random.Random(), "weights": self.fresh,
                 "cands": {}, "history": [dict(self.track0)], "error": None, "thread": None,
                 "method": GRPOStyle(self.policy, lr=1e-5, steps_per_round=8, choice_clip=0.2)}
            self.sessions[s["id"]] = s
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
            return {**out, "waited_seconds": round(time.time() - t, 1)}

        return api
