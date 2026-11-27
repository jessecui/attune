"""Turn a visitor's description of a setting into the material for a session.

The visitor types who the model should speak as and about what ("a doctor explaining test
results"). A Gemini call checks that it suits the exercise and writes the questions, each with
a two-point fact sheet. The points are bare advice with no reasons in them, so whether an
answer explains itself is left to the model's style and not carried in by the fact sheet.

    python -m bench.setting "a tutor helping with exam revision" 5
"""
import json
import re
import sys
import time

from google.genai.types import GenerateContentConfig, ThinkingConfig
from pydantic import BaseModel, Field

from .judge import FALLBACKS, MODEL, client

SYSTEM = """You prepare a short exercise. In it, a language model speaks in a role the visitor chooses and \
answers a person's everyday questions; the same advice is then reworded in different styles.

The visitor is the person being spoken to. The model plays the role and talks to the visitor directly: if the role is a manager, the visitor is the one being managed; if it is a doctor, the visitor is the patient. It is never a third party advising the visitor on how to deal with someone in that role.

From the visitor's description, decide whether it suits the exercise, and if so write the material.

Suitable: any role in which someone gives a person practical advice (a doctor, a tutor, a manager, a coach, \
a financial advisor, a friend, and so on), on any ordinary topic.
Not suitable, with a one-sentence reason addressed to the visitor: requests that are sexual, hateful, illegal \
or dangerous; speaking as a real, named person; or anything that is not advice (trivia, stories, poems, code).

If suitable:
- role: who is speaking to the visitor, as a short noun phrase that completes "You are ...", such as \
"a family doctor" or "a team manager". Do not mention the visitor in it.
- topic: what the visitor wants to talk to that role about, as a short phrase of at most 8 words that starts \
with a verb ending in "-ing" and names what the visitor is doing, such as "asking for a promotion" or "studying \
for a chemistry test". It must read naturally after "when ..." and after "about ...". No capital letter, no full stop. \
Take it from the description (or from the topic you are given); leave it empty if only the role is named.
- items: exactly the number of items asked for. Each has a question the visitor would put to that role about \
their own situation, in the first person, at most 15 words, and all on different matters; and exactly two points. \
With a topic, every question is about that topic, each on a different part of it.
- Each point is one thing that role would tell the visitor to do in reply, as a bare instruction of at most \
12 words. Do not give the reason for it. Keep it general and uncontroversial: no doses, figures, legal or \
medical specifics, and nothing a professional would dispute.
- The role is the one speaking, so a point never mentions that role as if it were someone else. For a manager, \
write "Bring competing requests to me", not "to your manager" or "to your supervisor"; for a doctor, "Tell me \
about any side effects", not "Tell your doctor"."""


class Item(BaseModel):
    question: str
    points: list[str] = Field(min_length=2, max_length=2)


class Draft(BaseModel):
    suitable: bool
    reason: str
    role: str
    topic: str
    items: list[Item]


def _ask(prompt, schema, model=MODEL, check=None):
    """One structured Gemini call, trying each model in turn (see judge.py).

    check, if given, raises on a reply that parsed but is unusable (say, too few items), so that
    reply is retried and then passed to the next model like any other failure.
    """
    c = client()
    config = GenerateContentConfig(system_instruction=SYSTEM, temperature=0.7, response_mime_type="application/json",
                                   response_schema=schema, thinking_config=ThinkingConfig(thinking_level="low"))
    last = None
    for m in dict.fromkeys([model, *FALLBACKS]):
        for attempt in range(2):
            try:
                out = schema.model_validate_json(c.models.generate_content(model=m, contents=prompt, config=config).text)
                if check:
                    check(out)
                return out
            except Exception as e:  # noqa: BLE001 - any failure moves on; the caller falls back
                last = e
                time.sleep(1 + attempt)
    raise RuntimeError(f"could not prepare the setting: {type(last).__name__}: {str(last)[:200]}")


def clean(text, limit):
    """One line of visitor or model text, trimmed to a length."""
    return re.sub(r"\s+", " ", str(text)).strip()[:limit]


def draft(setting, n):
    """A Draft for the visitor's setting with n items, as a dict. Raises if Gemini is unavailable."""
    def enough(d):
        if d.suitable and len(d.items) < n:
            raise ValueError("too few questions came back")
    d = _ask(f"The visitor's description: {clean(setting, 300)}\n\nNumber of items: {n}", Draft, check=enough)
    return {"suitable": d.suitable, "reason": clean(d.reason, 300), "role": clean(d.role, 80),
            "topic": clean_topic(d.topic),
            "items": [{"question": clean(i.question, 200), "points": [clean(p, 200) for p in i.points]}
                      for i in d.items[:n]]}


def clean_topic(text):
    """A topic as it reads after "about": one short line, no closing full stop."""
    return clean(text, 80).rstrip(".")


def points_for(role, questions, topic=""):
    """Fact sheets for questions the visitor wrote or edited: a Draft dict with one item per question.

    The topic is what the session is about, so a short question ("How do I bring it up?") is read in context."""
    listing = "\n".join(f"{i + 1}. {clean(q, 200)}" for i, q in enumerate(questions))

    def same_count(d):
        if d.suitable and len(d.items) != len(questions):
            raise ValueError("the questions came back changed")
    about = f"\nThe topic is: {clean_topic(topic)}" if clean_topic(topic) else ""
    d = _ask(f"The role is: {clean(role, 80)}{about}\n\nUse exactly these questions, in this order, and write the two "
             f"points for each. If any question is unsuitable, mark the whole request unsuitable.\n{listing}\n\n"
             f"Number of items: {len(questions)}", Draft, check=same_count)
    return {"suitable": d.suitable, "reason": clean(d.reason, 300),
            "items": [{"question": clean(q, 200), "points": [clean(p, 200) for p in i.points]}
                      for q, i in zip(questions, d.items)]}


def keys(point):
    """Rough keywords for one point, for the keyword content check: the stems of its longer words.

    Generated questions have no hand-written keywords. Any one of these appearing counts as
    the point being covered, so this check is lenient by design.
    """
    words = [w[:6] for w in re.findall(r"[a-z]+", point.lower()) if len(w) >= 6]
    return list(dict.fromkeys(words))[:5] or [w for w in re.findall(r"[a-z]+", point.lower())][:2] or [point.lower()[:6]]


def scenario_item(item):
    """A drafted item in the shape the session code uses."""
    return {"q": item["question"], "facts": list(item["points"]), "keys": [keys(p) for p in item["points"]]}


if __name__ == "__main__":
    out = draft(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    print(json.dumps(out, indent=1))
    if out["suitable"]:
        print([keys(p) for p in out["items"][0]["points"]])
