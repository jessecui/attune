"""Synthetic users: deterministic scoring functions with a hidden style preference.

A persona stands in for the human who ranks candidates. Because the preference is
known and fixed, a run is repeatable and we can measure whether tuning recovered it.
"""
import re

WARM_PHRASES = [
    "great question", "good question", "don't worry", "no worries", "let's",
    "you're", "you've", "you'll", "glad", "happy to", "wonderful", "awesome",
    "nice work", "exactly", "good news", "together", "feel free", "curious",
    "think of it", "imagine",
]


DIRECTIVE = ["you should", "you need to", "you must", "make sure", "be sure to", "start by",
             "always", "never", "i recommend", "the best thing", "right away", "immediately"]
COLLABORATIVE = ["you could", "you might", "you may", "one option", "another option", "options",
                 "consider", "if you prefer", "up to you", "might help", "may help", "could help",
                 "some people", "whichever", "works for you", "you can also", "alternatively", "either"]
IMPERATIVE_VERBS = set("""set put write keep pay read send list start track bring drink cancel report check
ask get use make take tell rank rehearse gather compare wait adjust stand spread test summarize explain
link aim give research try stop turn move go do place schedule create build practice speak focus
prioritize separate know leave end look raise tie treat""".split())
EMPATHY = ["i understand", "i know", "that sounds", "it sounds", "understandable", "sorry to hear",
           "it can be hard", "it's hard", "can be tough", "frustrating", "stressful", "overwhelming",
           "it's normal", "it's common", "you're not alone", "i hear you", "hi there", "hello"]
REASONS = ["because", "so that", "since ", "which helps", "which makes", "this way", "that way", "as it ",
           "so you", "to help", "to make", "to keep", "to avoid", "to reduce", "this helps", "this makes",
           "the reason", "that's why", "which means"]
ENERGETIC = ["great", "really", "absolutely", "exciting", "fantastic", "awesome", "love", "amazing",
             "you've got this", "you can do"]


def _clamp(x):
    return max(0.0, min(1.0, x))


def features(text):
    t = text.lower()
    words = re.findall(r"[a-zA-Z0-9^'’]+", t)
    n = len(words)
    warm_hits = sum(t.count(p) for p in WARM_PHRASES) + t.count("!")
    long_frac = sum(len(w) >= 9 for w in words) / max(n, 1)
    sentences = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
    commands = sum(s.split()[0] in IMPERATIVE_VERBS for s in sentences)
    directive = sum(t.count(p) for p in DIRECTIVE) + commands
    collaborative = sum(t.count(p) for p in COLLABORATIVE)
    return {
        "sentences": len(sentences),
        "short": _clamp((45 - n) / 25),
        "warm_words": _clamp(sum(t.count(p) for p in WARM_PHRASES + EMPATHY) / 2),
        "directive": _clamp(0.5 + (directive - collaborative) / 4),
        "energy": _clamp((t.count("!") + sum(t.count(p) for p in ENERGETIC)) / 3),
        "reasons": _clamp(sum(t.count(p) for p in REASONS) / 2),
        "casual": _clamp(len(re.findall(r"\b\w+['’](?:ll|re|ve|s|d|t|m)\b", t)) / 2),
        "words": n,
        "warmth": _clamp(warm_hits / 3),
        "technical": _clamp(long_frac / 0.25),
        "ends_question": 1.0 if text.rstrip().endswith("?") else 0.0,
    }


def warm_brief(text):
    """Likes short, friendly answers that end with a check-in question."""
    f = features(text)
    # Peaks at 30-50 words: a real user would not reward a cut-off fragment.
    brevity = min(_clamp((f["words"] - 10) / 20), _clamp((140 - f["words"]) / (140 - 50)))
    return brevity + f["warmth"] + 0.5 * f["ends_question"]


def thorough_formal(text):
    """Likes long, neutral answers with technical vocabulary."""
    f = features(text)
    length = _clamp((f["words"] - 40) / (120 - 40))
    return length + (1 - f["warmth"]) + 0.5 * f["technical"]


def by_axes(**signs):
    """A persona that likes (+1) or dislikes (-1) each style feature. Score in [-5, 5]."""
    def score(text):
        f = features(text)
        return sum(sign * (2 * f[name] - 1) for name, sign in signs.items())
    score.signs = signs
    return score


# What each feature reads, on average, for an axis's two options (7B model, current menu wording;
# from the menu check). Used to put the four live axes on one scale, so that a simulated user
# cares about each axis equally even though the word lists are not equally sensitive.
LIVE_RANGE = {"warm_words": (0.06, 0.38), "directive": (0.53, 0.81), "casual": (0.04, 0.35), "reasons": (0.08, 0.46)}


def by_live_axes(**signs):
    """A persona over the four live axes: +1 likes the axis's first option, -1 its second."""
    def score(text):
        f = features(text)
        total = 0.0
        for name, sign in signs.items():
            lo, hi = LIVE_RANGE[name]
            total += sign * max(-2.0, min(2.0, (f[name] - (lo + hi) / 2) / ((hi - lo) / 2)))
        return total
    score.signs = signs
    return score


PERSONAS = {
    "warm_brief": warm_brief,
    "thorough_formal": thorough_formal,
    # For the five-axis menu (data/advice.json): one preferred end per axis.
    "gentle_guide": by_axes(short=-1, warm_words=+1, directive=-1, technical=-1, energy=-1),
    "brisk_expert": by_axes(short=+1, warm_words=-1, directive=+1, technical=+1, energy=-1),
    "upbeat_coach": by_axes(short=+1, warm_words=+1, directive=+1, technical=-1, energy=+1),
    # For the four live axes (warmth, directiveness, formality, depth). Each axis is liked
    # by two of the four and disliked by the other two.
    "gentle_explainer": by_live_axes(warm_words=+1, directive=-1, casual=+1, reasons=+1),
    "blunt_friend": by_live_axes(warm_words=-1, directive=+1, casual=+1, reasons=-1),
    "kind_professional": by_live_axes(warm_words=+1, directive=+1, casual=-1, reasons=-1),
    "reserved_analyst": by_live_axes(warm_words=-1, directive=-1, casual=-1, reasons=+1),
}


def rank(persona, texts):
    """Indices of texts from most to least preferred (ties keep input order)."""
    score = PERSONAS[persona]
    return sorted(range(len(texts)), key=lambda i: -score(texts[i]))


def facts_retained(text, keys):
    """Rough check: share of fact-sheet points with at least one key term present."""
    t = text.lower()
    return sum(any(k in t for k in group) for group in keys) / len(keys)
