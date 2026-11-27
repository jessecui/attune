"""A live Attune session on Modal: a real person gives the feedback.

One GPU container holds the model and every open session's adapter. A plain test
page is served from the same app; the real interface will live elsewhere.

    modal deploy modal_live.py        # prints the URLs; requests need a Modal proxy token and ?key=<contents of .live_key>

Each round the page shows four short answers and the person marks the best and the
worst. The four are written under four different style instructions, which are there
only to make them differ: the training is aimed at the plain prompt, so the best pick
becomes more likely, and the worst less likely, as an answer to the question with no
style named. The finished model is never given a style.

To hide the wait, the server works one round ahead: while the person reads round N,
it applies their round N-1 feedback and then writes round N+1's answers. So a round's
feedback first shapes the answers two rounds later. Nothing is measured between
rounds: at the end the tuned model answers unseen questions from the plain prompt and
a judge model rates them (see _summary).
"""
import secrets as pysecrets
from pathlib import Path

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
AXES = ["warmth", "directiveness", "formality", "depth"]
# Seven questions, trained with DPO. Over both sets of benchmark runs (grids "long" and "confirm") its win rate
# over the untouched model was 93% after five rounds and 95% after six, seven and eight, with the style fit still
# rising a little each round; from ten rounds it starts to drop facts. Its text stays intact at every length (1 to
# 2% malformed), where GRPO with replay, the method compared against it, reached 24% malformed after ten rounds.
ROUNDS, K = 7, 4
MEASURE = 5   # unseen questions, answered and rated once at the end; all of them are shown as before and after
SHOWN = dict(temperature=0.7, top_p=0.95)   # how the answers on the results screen are written
# A session lasts this long from the moment its first question is ready, whatever the visitor is doing:
# questions, results and the try box. It caps what one session can cost. The page shows the time left.
SESSION_SECONDS = 20 * 60
SESSION_GRACE = 60   # a pick sent in the last seconds is still taken
EXPIRED = "Your 20 minutes are up. Train another to start again."
# The rate both methods were compared at: the highest of the three tried at which neither lost content
# (at 1e-4 a third of answers dropped a fact, for both).
LEARNING_RATE = 2e-5
# What the visitor sees when background work failed: the session cannot go on (its answers are spent).
STUCK = "Something went wrong while training. Please train another to start again."

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

PACKAGE_README = """YOUR ATTUNE MODEL

Trained on {date} from your picks in a {rounds}-question Attune session.


WHAT'S IN THIS FOLDER

  adapter_model.safetensors, adapter_config.json
      Your model's trained part: the changes from your session. It runs on top of
      the standard {base} model, which the script downloads.
  attune.json
      Your results, and the settings your model was trained with.
  example.py
      A script that asks your model a question.


WHAT YOU NEED

  - A Mac with Apple silicon and at least 24 GB of memory (32 GB or more runs
    comfortably), or a computer with a graphics card that has at least 16 GB.
  - Python 3.10 or later.
  - About 16 GB of free disk space. The first run downloads the standard model
    (about 15 GB).


HOW TO RUN IT

  1. Open a terminal in this folder.

  2. Set up Python and install what the script needs (a few minutes):

       python3 -m venv .venv
       .venv/bin/pip install torch transformers peft accelerate

  3. Quit your browser, code editors, and other large apps first: with 24 GB of
     memory, the model uses most of it. Then ask your model a question:

       .venv/bin/python example.py "How do I ask for a raise?"

     The first run downloads the standard model before it answers. After that,
     it takes about a minute to load each time.

  On Windows, use .venv\\Scripts\\ in place of .venv/bin/.


GOOD TO KNOW

  - It was trained on short pieces of advice, so that is where it sounds most
    like your session.
  - In your session every answer was given two points to make. Asked a bare
    question, it answers from what it knows. To give it points, add them after
    the question:
      .venv/bin/python example.py "How do I ask for a raise?" "Gather evidence of your results" "Research typical pay"
  - It knows no more than the standard model. Its answers show a style; they can
    be wrong and are not medical, legal, or financial advice.
  - To free the disk space afterwards, delete this folder:
      ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
"""

# The part of the system prompt that holds an answer to its fact sheet.
POINTS_RULE = "Base your advice only on the points provided and cover both of them. You may add a brief remark that sets the tone, but no new advice. "

PACKAGE_EXAMPLE = """\"\"\"Ask your Attune model a question: python example.py "How do I ask for a raise?"

To give it the points to make, as in your session, add them after the question:
    python example.py "How do I ask for a raise?" "Gather evidence of your results" "Research typical pay"
\"\"\"
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
base = AutoModelForCausalLM.from_pretrained(meta["base_model"], dtype=torch.float32 if device == "cpu" else torch.bfloat16)  # 16-bit on a GPU or Mac: about 15 GB
model = PeftModel.from_pretrained(base, str(here)).to(device).eval()

question = sys.argv[1] if len(sys.argv) > 1 else "How do I ask for a raise?"
points = sys.argv[2:]
# No style is named anywhere: the style is in the adapter's weights. With points, the prompt is the one
# your session used (the answer is held to them); without, the model answers from what it knows.
if points:
    system, user = meta["system_prompt_with_points"], f"Question: {question}\\n\\nFacts:\\n" + "\\n".join(f"- {p}" for p in points)
else:
    system, user = meta["system_prompt"], f"Question: {question}"
messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
# return_dict=True works across transformers versions (newer ones return a mapping, not a bare tensor)
enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
# a little randomness, so answers do not all open with the same phrase
out = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.95)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
"""


def key_ok(given, expected):
    import hmac
    return bool(expected) and hmac.compare_digest(given.encode(), expected.encode())


@app.cls(image=image, gpu=["L40S", "A100-40GB", "A100-80GB"], volumes={"/cache": hf_cache, "/adapters": adapters}, max_containers=1,
         scaledown_window=300,
         # the judge model's key, for rating the tuned model's writing at the end
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
        self.args = argparse.Namespace(two_step=False, k=K, max_new_tokens=128, max_sentences=4,
                                       min_facts=1.0, strict_facts=True, redraws=3,
                                       # a near-copy of another answer is rewritten in a style not on screen
                                       no_duplicates=True)
        self.policy = Policy(MODEL)
        self.fresh = self.policy.snapshot()  # the untouched adapter every session starts from
        self.sessions = {}
        hf_cache.commit()

    def _open(self, s):
        """Set a new session up on its own questions: prompts, and the standard model's answers to
        the rating questions (written here, rated in the background) as the "before"."""
        import threading
        s["sess"] = self.R.Session(self.policy, s["scenario"], self.args, latent=False)
        # Sampled, not the single most likely wording: the same setting as the tuned model's answers below.
        items = s["scenario"]["heldout"]
        outs = self.policy.generate(s["sess"].plain["heldout"], max_new_tokens=self.args.max_new_tokens,
                                    use_adapter=False, **SHOWN)
        s["base_texts"] = s["sess"].finish(outs, items=items)[1]
        s["written0"] = None
        s["rating0"] = threading.Thread(
            target=lambda: s.update(written0=self._written(s["scenario"]["heldout"], s["base_texts"])))
        s["rating0"].start()
        s["groups"] = []   # every round's answers and picks: each update also revisits earlier rounds

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

        None if the judge is unavailable; the page then shows the answers without ratings.
        """
        try:
            from bench.judge import judge_many
            verdicts = judge_many([(it["q"], it["facts"], t) for it, t in zip(items, texts)])
            return {a: round(sum(v[a] for v in verdicts) / len(verdicts), 2) for a in AXES}
        except Exception as e:  # noqa: BLE001 - a judge outage must not end the session
            print(f"judge unavailable: {e}", flush=True)
            return None

    def _final(self, s):
        """The session's model on the rating questions: the plain prompt, with no style in it."""
        sess, items = s["sess"], s["scenario"]["heldout"]
        # Sampled with a little randomness, as the standard model's answers were: taking the single
        # most likely wording every time makes answers open with the same phrase.
        outs = self.policy.generate(sess.plain["heldout"], max_new_tokens=self.args.max_new_tokens, **SHOWN)
        return sess.finish(outs, items=items)[1]

    def _write(self, s, n):
        """Write round n's candidates with the session's current adapter."""
        import time
        t = time.time()
        i = (n - 1) % len(s["scenario"]["train"])
        # Four different style instructions, drawn at random: they only make the four answers differ.
        picks, answers, texts, redrawn, _ = s["sess"].propose(i, s["rng"])
        s["cands"][n] = {"answers": answers, "prompt": s["sess"].plain["train"][i],
                         "payload": {"session": s["id"], "round": n, "rounds": s["rounds"],
                                     "question": s["scenario"]["train"][i]["q"], "candidates": texts,
                                     "role": s["role"], "topic": s.get("topic", ""), "title": s.get("title", ""),
                                     "builtin": s["builtin"],
                                     "notice": s["notice"],
                                     "write_seconds": round(time.time() - t, 1)}}

    def _learn(self, s, n, best, worst):
        """Apply round n's feedback to the session's adapter."""
        from bench.methods import Group
        c = s["cands"].pop(n)
        # Each answer is credited to the plain prompt, not to the style instruction it was written under.
        order = [best, *[j for j in range(K) if j not in (best, worst)], worst]
        # DPO measures each answer's change against the untouched model, so that model's score is taken once here.
        ref = self.policy.logprobs([c["prompt"]] * K, c["answers"], use_adapter=False).detach().cpu()
        s["groups"].append(Group(c["prompt"], c["answers"], order, ref, None, bestworst=True))
        s["method"].update(s["groups"], None)
        # Nothing is measured between rounds: the model is rated once, at the end (see _summary).

    def _after(self, s, n, best, worst):
        """Background work after round n's feedback: learn from it, then write round n+2."""
        def work():
            self._with(s, lambda: self._learn(s, n, best, worst))
            if n + 2 <= s["rounds"]:
                self._with(s, lambda: self._write(s, n + 2))
        self._background(s, work)

    def _background(self, s, work):
        """Run a piece of a session's work; a failure is kept and surfaced on the next request."""
        try:
            work()
        except Exception as e:  # noqa: BLE001
            s["error"] = repr(e)[:300]
            print(f"session {s['id']}: {s['error']}", flush=True)   # the detail goes to the log, not the visitor

    def _summary(self, s):
        # Measured once, at the end: the tuned model answers the unseen questions from the plain prompt
        # and the judge rates each scale. The standard model's ratings on the same questions (taken at
        # start-up) are the comparison. Only the answering needs the GPU; the judge calls run outside
        # the lock, so a slow Gemini does not hold up other visitors' sessions.
        texts = self._with(s, lambda: self._final(s))
        s["rating0"].join(timeout=40)
        written = [s["written0"], self._written(s["scenario"]["heldout"], texts)]
        rated = all(written)
        axes = [{"axis": m["axis"], "first": m["options"][0][0], "second": m["options"][1][0],
                 "written": [w[m["axis"]] for w in written] if rated else None} for m in self.menu]
        examples = [{"question": it["q"], "before": b, "after": t}
                    for it, b, t in zip(s["scenario"]["heldout"], s["base_texts"], texts)]
        return {"done": True, "session": s["id"], "rounds": s["rounds"], "axes": axes, "role": s["role"],
                "topic": s.get("topic", ""), "title": s.get("title", ""), "builtin": s["builtin"], "examples": examples}

    def _try(self, s, item):
        """One question of the visitor's own, answered by the standard model and by their model.

        Both get the same prompt: the question and a two-point fact sheet (written for it just before
        this), and no style. The only difference is the session's adapter.
        """
        sess = s["sess"]
        plain = self.policy.prompt_ids(s["scenario"]["system"], self.R.user_message(item))
        try:   # both answers in one pass on the GPU
            base, tuned = ([o] for o in self.policy.generate([plain, plain], max_new_tokens=self.args.max_new_tokens,
                                                              adapters=[False, True], **SHOWN))
        except Exception as e:  # noqa: BLE001 - fall back to one after the other
            print(f"mixed batch failed, writing in turn: {e}", flush=True)
            base = self.policy.generate([plain], max_new_tokens=self.args.max_new_tokens, use_adapter=False, **SHOWN)
            tuned = self.policy.generate([plain], max_new_tokens=self.args.max_new_tokens, **SHOWN)
        return {"question": item["q"], "before": sess.finish(base, items=[item])[1][0],
                "after": sess.finish(tuned, items=[item])[1][0]}

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
        (d / "README.md").unlink(missing_ok=True)   # PEFT's empty model card; instructions.txt is the guide
        meta = {"base_model": MODEL, "created": time.strftime("%Y-%m-%d"), "rounds": s["rounds"],
                # the judge's ratings of the writing, 1 to 5 toward "first": the standard model's, then this one's
                "profile": [{"axis": a["axis"], "first": a["first"], "second": a["second"],
                             "standard_model": a["written"][0] if a["written"] else None,
                             "this_model": a["written"][-1] if a["written"] else None} for a in summary["axes"]],
                "speaking_as": s["role"], "about": s.get("topic", ""), "questions": [it["q"] for it in s["scenario"]["train"][: s["rounds"]]],
                # In the session every question came with two points to make, and the system prompt says so.
                # Asked a bare question there are no points, so that sentence is left out.
                "system_prompt": s["scenario"]["system"].replace(POINTS_RULE, ""),
                "system_prompt_with_points": s["scenario"]["system"],
                "prompt_format": "The user turn is 'Question: ...'. With points to make, it continues with 'Facts:' and "
                                 "a bulleted list, and the system prompt is system_prompt_with_points. No style is "
                                 "named anywhere: the adapter carries the style."}
        (d / "attune.json").write_text(json.dumps(meta, indent=1))
        (d / "instructions.txt").write_text(PACKAGE_README.format(base=MODEL, date=meta["created"], rounds=s["rounds"]))
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

    @modal.asgi_app(requires_proxy_auth=True)   # Modal refuses a request without the proxy token before a container starts
    def web(self):
        import os
        import random
        import threading
        import time
        import uuid

        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel

        from bench.methods import DPO

        api = FastAPI()

        def check(key):
            if not key_ok(key, os.environ.get("ATTUNE_KEY", "")):
                raise HTTPException(status_code=403, detail="Wrong or missing key")

        def seconds_left(s):
            """What the page shows: the time left of the session's twenty minutes."""
            return max(0, int(SESSION_SECONDS - (time.time() - s["started"])))

        def in_time(s):
            """End a session that has run past its limit (and the few seconds of grace)."""
            if time.time() - s["started"] > SESSION_SECONDS + SESSION_GRACE:
                self.sessions.pop(s["id"], None)
                raise HTTPException(status_code=410, detail=EXPIRED)

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
            topic: str = ""   # what the visitor wants to talk about; empty when they named only the role
            title: str = ""   # the setting in one line, as the page shows it above each question
            items: list[DraftItem]
            measure: list[DraftItem]

        class StartIn(BaseModel):
            rounds: int | None = None
            draft: DraftIn | None = None

        def scenario_from(draft, rounds):
            """The session's questions from a visitor's draft; fact sheets are written here for
            any question they typed themselves. Returns (scenario, role, topic, title)."""
            from bench import setting
            role = setting.clean(draft.role, 80).rstrip(".") or "an advisor"
            topic, title = setting.clean_topic(draft.topic), setting.clean_topic(draft.title)
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
                    written = setting.points_for(role, [i["question"] for i in typed], topic)
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
                opening, f"You are {role}. The person has come to you with a question{f' about {topic}' if topic else ''}. "
                         f"Answer them directly, speaking as yourself in that role.", 1)
            return {"system": system, "train": made[:rounds], "heldout": made[rounds:]}, role, topic, title

        @api.post("/api/start")
        def start(body: StartIn | None = None, key: str = "", rounds: int = ROUNDS):
            check(key)
            rounds = (body and body.rounds) or rounds
            if rounds != ROUNDS:
                raise HTTPException(status_code=400, detail=f"Rounds must be {ROUNDS}")
            scenario, role, topic, title, builtin = self.builtin, "an advisor", "", "", True
            if body and body.draft:
                scenario, role, topic, title = scenario_from(body.draft, rounds)
                builtin = False
            # Sessions past their time limit, or left before their first question was ready, are dropped.
            for sid in [k for k, old in list(self.sessions.items())
                        if time.time() - old.get("started", old["touched"]) > SESSION_SECONDS + SESSION_GRACE]:
                self.sessions.pop(sid, None)   # another start may have dropped it already
            s = {"id": uuid.uuid4().hex, "round": 0, "rounds": rounds, "rng": random.Random(), "weights": self.fresh,
                 "touched": time.time(),
                 "scenario": scenario, "role": role, "topic": topic, "title": title, "builtin": builtin, "notice": None,
                 "cands": {}, "error": None, "thread": None,
                 # DPO on the plain prompt, as in the benchmark: each pick gives five preferred-over pairs,
                 # and every other step of an update revisits an earlier round's pick.
                 "method": DPO(self.policy, lr=LEARNING_RATE, steps_per_round=8, seed=random.randrange(2 ** 31))}
            self.sessions[s["id"]] = s
            self._with(s, lambda: self._open(s))
            self._with(s, lambda: self._write(s, 1))
            if rounds >= 2:  # get one round ahead straight away
                s["thread"] = threading.Thread(target=self._background,
                                               args=(s, lambda: self._with(s, lambda: self._write(s, 2))))
                s["thread"].start()
            s["started"] = time.time()   # the clock starts now, not while the GPU was starting
            return {**s["cands"][1]["payload"], "seconds_left": seconds_left(s)}

        class Feedback(BaseModel):
            session: str
            best: int
            worst: int
            round: int | None = None   # the question being answered; lets a resent submit be recognised

        @api.post("/api/feedback")
        def feedback(fb: Feedback, key: str = ""):
            check(key)
            s = self.sessions.get(fb.session)
            if s is None:
                raise HTTPException(status_code=404, detail="Session expired; start again")
            s["touched"] = time.time()
            if "started" not in s:
                raise HTTPException(status_code=409, detail="The first question is still being written")
            if s.get("finished"):
                # A repeat of the last submit (say the reply was lost on the way): send the results again.
                return {**s["summary"], "waited_seconds": 0.0, "seconds_left": seconds_left(s)}
            in_time(s)
            if fb.round is not None and fb.round == s["round"] < s["rounds"] and s.get("reply"):
                return s["reply"]   # this question's picks were already applied: resend what came back
            if fb.round is not None and fb.round != s["round"] + 1 and s["round"] < s["rounds"]:
                # An older question arriving late (or one never shown): applying it would train on
                # answers the visitor did not pick from.
                raise HTTPException(status_code=409, detail="That question has already been answered")
            if fb.best == fb.worst or not {fb.best, fb.worst} <= set(range(K)):
                raise HTTPException(status_code=400, detail="Pick two different answers")
            # One answer at a time per session: a repeated submit would otherwise count as the next round.
            busy = s.setdefault("busy", threading.Lock())
            if not busy.acquire(blocking=False):
                raise HTTPException(status_code=409, detail="Still working on your last answer")
            try:
                t = time.time()
                if s["thread"]:
                    s["thread"].join()  # wait for the work started after the previous round
                if s["error"]:
                    raise HTTPException(status_code=500, detail=STUCK)
                if s["round"] < s["rounds"]:   # else the last update is done and only the results failed: retry those
                    n = s["round"] = s["round"] + 1
                    if n < s["rounds"]:
                        payload = {**s["cands"][n + 1]["payload"], "waited_seconds": round(time.time() - t, 1),
                                   "seconds_left": seconds_left(s)}
                        s["thread"] = threading.Thread(target=self._after, args=(s, n, fb.best, fb.worst))
                        s["thread"].start()
                        s["reply"] = payload
                        return payload
                    self._after(s, n, fb.best, fb.worst)  # last round: nothing left to hide the wait behind
                    if s["error"]:
                        raise HTTPException(status_code=500, detail=STUCK)
                out = self._summary(s)
                s["summary"] = out
                # The finished session stays (until it has been idle for half an hour, or the GPU shuts
                # down) so the visitor can try their model on questions of their own.
                s["finished"] = True
                # Package the adapter in the background; the download appears a few seconds later.
                threading.Thread(target=lambda: self._with(s, lambda: self._export(s, out))).start()
                return {**out, "waited_seconds": round(time.time() - t, 1), "seconds_left": seconds_left(s)}
            finally:
                busy.release()

        class TryIn(BaseModel):
            session: str
            question: str

        def finished(session):
            s = self.sessions.get(session)
            if s is None or not s.get("finished"):
                raise HTTPException(status_code=404, detail="This model is no longer loaded. Train another to try more questions.")
            in_time(s)
            s["touched"] = time.time()
            return s

        def prepare_question(s, question):
            """A fact sheet for the visitor's question (no GPU needed), as for every other question in
            the app: without one the two models give different advice, and the tuned model, which
            was trained on fact sheets, drifts out of its role."""
            from bench import setting
            question = " ".join(question.split())[:300]
            if len(question) < 5:
                raise HTTPException(status_code=400, detail="Type a question first")
            try:
                written = setting.points_for(s["role"], [question], s.get("topic", ""))
            except Exception:  # noqa: BLE001
                raise HTTPException(status_code=502, detail="Could not prepare that question just now. Try again.")
            if not written["suitable"]:
                raise HTTPException(status_code=400, detail=written["reason"] or "That question is not suitable")
            return setting.scenario_item(written["items"][0])

        # The page calls these two in turn, so it can say which step it is waiting on.
        @api.post("/api/try/prepare")
        def try_prepare(body: TryIn, key: str = ""):
            check(key)
            s = finished(body.session)
            s["try_item"] = prepare_question(s, body.question)
            return {"prepared": True}

        @api.post("/api/try/answer")
        def try_answer(body: TryIn, key: str = ""):
            check(key)
            s = finished(body.session)
            item = s.pop("try_item", None)
            if item is None:
                raise HTTPException(status_code=400, detail="Ask the question again")
            return self._with(s, lambda: self._try(s, item))

        @api.post("/api/try")   # both steps in one call
        def try_model(body: TryIn, key: str = ""):
            check(key)
            s = finished(body.session)
            item = prepare_question(s, body.question)
            return self._with(s, lambda: self._try(s, item))

        return api


@app.function(image=image, volumes={"/adapters": adapters},
              timeout=1800,   # a 160 MB download on a slow connection outlasts the default five minutes
              secrets=[modal.Secret.from_dict({"ATTUNE_KEY": KEY}), modal.Secret.from_name("attune-judge")])
@modal.asgi_app(requires_proxy_auth=True)
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
        if body.rounds != ROUNDS:
            raise HTTPException(status_code=400, detail=f"Rounds must be {ROUNDS}")
        os.chdir("/root")
        from bench import setting
        if len(setting.clean(body.setting, 300)) < 3:
            raise HTTPException(status_code=400, detail="Say who the model should speak as")
        try:
            d = setting.draft(body.setting, body.rounds + MEASURE)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=502, detail="Could not prepare that setting just now")
        return {"suitable": d["suitable"], "reason": d["reason"], "role": d["role"], "topic": d["topic"], "title": d["title"],
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

