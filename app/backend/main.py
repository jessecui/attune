"""Attune's web backend: a password gate in front of the GPU service on Modal.

The browser only ever talks to this server. The Modal URLs and key stay here, so the
GPU endpoint is never exposed to visitors.

    uvicorn main:app --reload --port 8787        # from app/backend, with the env vars below

Environment:
    CREATOR_NAME        the name the access prompt asks for (required); matched ignoring case and spaces
    ATTUNE_SECRET       signs the session cookie (optional; derived from the other settings if unset)
    ATTUNE_MODAL_URL    the live session service, e.g. https://...-live-web.modal.run
    ATTUNE_FILES_URL    the download service, e.g. https://...-files.modal.run
    ATTUNE_MODAL_KEY    the key both Modal services expect
    ATTUNE_MOCK=1       answer from a recorded session instead of calling Modal (no GPU, no cost)
"""
import asyncio
import hashlib
import hmac
import json
import math
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

def normalise(name: str) -> str:
    """A name with case and all whitespace removed, so "Ada Lovelace", "ada lovelace" and " ADALOVELACE " match."""
    return "".join(name.split()).casefold()


CREATOR = normalise(os.environ.get("CREATOR_NAME", ""))
MODAL_URL = os.environ.get("ATTUNE_MODAL_URL", "").rstrip("/")
FILES_URL = os.environ.get("ATTUNE_FILES_URL", "").rstrip("/")
MODAL_KEY = os.environ.get("ATTUNE_MODAL_KEY", "")
MOCK = os.environ.get("ATTUNE_MOCK") == "1"
# The cookie-signing secret. If none is set it is derived from the name and the GPU key, so it
# survives restarts and changing either signs everyone out.
SECRET = os.environ.get("ATTUNE_SECRET") or hashlib.sha256(f"attune-session|{CREATOR}|{MODAL_KEY}".encode()).hexdigest()
COOKIE, COOKIE_SECONDS = "attune_session", 12 * 3600
# One session length keeps the app simple and matches what the benchmark tested. The GPU service
# still accepts 10 for experiments.
ROUND_CHOICES = (5,)

app = FastAPI(title="Attune")


# ---- password gate ---------------------------------------------------------------------

def sign(expires: int) -> str:
    return hmac.new(SECRET.encode(), str(expires).encode(), hashlib.sha256).hexdigest()


def signed_in(token: str | None) -> bool:
    try:
        expires, sig = (token or "").split(".")
        return int(expires) > time.time() and hmac.compare_digest(sig, sign(int(expires)))
    except ValueError:
        return False


def require(token: str | None):
    if not signed_in(token):
        raise HTTPException(status_code=401, detail="Please sign in")


class Login(BaseModel):
    answer: str


@app.post("/api/login")
def login(body: Login, request: Request, response: Response):
    if not CREATOR:
        raise HTTPException(status_code=500, detail="CREATOR_NAME is not set on the server")
    if not hmac.compare_digest(normalise(body.answer).encode(), CREATOR.encode()):
        time.sleep(0.5)  # slow down guessing
        raise HTTPException(status_code=401, detail="That name doesn't match. Please try again.")
    expires = int(time.time()) + COOKIE_SECONDS
    response.set_cookie(COOKIE, f"{expires}.{sign(expires)}", max_age=COOKIE_SECONDS, httponly=True,
                        samesite="lax", secure=request.url.scheme == "https")
    return {"signed_in": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"signed_in": False}


@app.get("/api/me")
def me(attune_session: str | None = Cookie(default=None)):
    return {"signed_in": signed_in(attune_session), "mock": MOCK, "round_choices": ROUND_CHOICES}


# ---- the session: forwarded to Modal, or replayed from a recording in mock mode --------------

async def modal(path: str, params: dict | None = None, body: dict | None = None):
    if not (MODAL_URL and MODAL_KEY):
        raise HTTPException(status_code=500, detail="The GPU service is not configured on the server")
    try:
        # Generous timeout: the first request may wait for the GPU to start and the model to load.
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post(f"{MODAL_URL}{path}", params={"key": MODAL_KEY, **(params or {})}, json=body)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach the GPU service ({type(e).__name__})")
    if r.status_code >= 400:
        detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else None
        raise HTTPException(status_code=502 if r.status_code >= 500 else r.status_code,
                            detail=detail or "The GPU service returned an error")
    return r.json()


_mock_sessions: dict[str, dict] = {}


def _mock():
    return json.loads((Path(__file__).parent / "mock_data.json").read_text())


def _mock_round(s, n):
    r = _mock()["rounds"][n - 1]
    return {"session": s["id"], "round": n, "rounds": s["rounds"], **r, "role": "an advisor", "builtin": True}


def _mock_summary(s):
    fx, n = _mock(), s["rounds"]
    # Soften the recording's leans so the mock profile shows every kind of result.
    scale = {"warmth": 0.45, "directiveness": 0.2, "formality": 0.15, "depth": 0.6}
    axes = [{**a, "lean": [round(v * scale[a["axis"]], 2) for v in fx["lean"][a["axis"]][: n + 1]]} for a in fx["axes"]]
    for a in axes:  # stand-in ratings of the writing, 1 to 5, so the mock looks like a real session
        a["written"] = [round(3 + 2 * math.tanh(v / 3), 1) for v in a["lean"]]
    return {"done": True, "session": s["id"], "rounds": n, "axes": axes, "examples": fx["examples"],
            "role": "an advisor", "builtin": True,
            "style_instructions": fx["style_instructions"], "tracked_question": fx["tracked_question"], "track": []}


class DraftItem(BaseModel):
    question: str
    points: list[str] | None = None   # None when the visitor wrote or edited the question


class Draft(BaseModel):
    role: str
    items: list[DraftItem]
    measure: list[DraftItem]


class Start(BaseModel):
    rounds: int = ROUND_CHOICES[0]
    draft: Draft | None = None   # the visitor's setting and questions; without it, the built-in questions


class Prepare(BaseModel):
    setting: str
    rounds: int = ROUND_CHOICES[0]


async def _warm():
    """Start the GPU container while the visitor reads their questions. Best effort."""
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            await client.get(f"{MODAL_URL}/api/warm", params={"key": MODAL_KEY})
    except httpx.HTTPError:
        pass


@app.post("/api/session/prepare")
async def prepare(body: Prepare, attune_session: str | None = Cookie(default=None)):
    """Check the visitor's setting and draft questions for it. No GPU is needed for this step."""
    require(attune_session)
    if body.rounds not in ROUND_CHOICES:
        raise HTTPException(status_code=400, detail=f"Rounds must be one of {ROUND_CHOICES}")
    if MOCK:
        await asyncio.sleep(1.2)
        fx = _mock()
        return {"suitable": True, "reason": "", "role": "an advisor (demo data ignores the setting)",
                "items": [{"question": r["question"], "points": ["", ""]} for r in fx["rounds"][: body.rounds]],
                "measure": [{"question": e["question"], "points": ["", ""]} for e in fx["examples"]] * 2}
    if not (FILES_URL and MODAL_KEY):
        raise HTTPException(status_code=500, detail="The setting service is not configured on the server")
    if MODAL_URL:
        asyncio.create_task(_warm())
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(f"{FILES_URL}/prepare", params={"key": MODAL_KEY}, json=body.model_dump())
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach the setting service ({type(e).__name__})")
    if r.status_code >= 400:
        detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else None
        raise HTTPException(status_code=502 if r.status_code >= 500 else r.status_code,
                            detail=detail or "Could not prepare that setting")
    return r.json()


class Feedback(BaseModel):
    session: str
    best: int
    worst: int


@app.post("/api/session/start")
async def start(body: Start, attune_session: str | None = Cookie(default=None)):
    require(attune_session)
    if body.rounds not in ROUND_CHOICES:
        raise HTTPException(status_code=400, detail=f"Rounds must be one of {ROUND_CHOICES}")
    if MOCK:
        await asyncio.sleep(1.5)
        s = {"id": uuid.uuid4().hex, "rounds": body.rounds, "round": 0}
        _mock_sessions[s["id"]] = s
        return _mock_round(s, 1)
    return await modal("/api/start", body=body.model_dump())


@app.post("/api/session/feedback")
async def feedback(body: Feedback, attune_session: str | None = Cookie(default=None)):
    require(attune_session)
    if MOCK:
        s = _mock_sessions.get(body.session)
        if s is None:
            raise HTTPException(status_code=404, detail="Session expired; start again")
        await asyncio.sleep(0.4)
        s["round"] += 1
        return _mock_summary(s) if s["round"] >= s["rounds"] else _mock_round(s, s["round"] + 1)
    return await modal("/api/feedback", body=body.model_dump())


class Try(BaseModel):
    question: str


@app.post("/api/session/{session}/try")
async def try_model(session: str, body: Try, attune_session: str | None = Cookie(default=None)):
    """A question of the visitor's own, answered by the standard model and by the model they just trained."""
    require(attune_session)
    _valid(session)
    if MOCK:
        await asyncio.sleep(1.0)
        ex = _mock()["examples"][0]
        return {"question": body.question, "before": ex["before"], "after": ex["after"]}
    return await modal("/api/try", body={"session": session, "question": body.question})


# ---- the tuned model, as a download ----------------------------------------------------------

def _valid(session: str):
    if len(session) != 32 or any(c not in "0123456789abcdef" for c in session):
        raise HTTPException(status_code=404, detail="No such session")


@app.get("/api/session/{session}/model/status")
async def model_status(session: str, attune_session: str | None = Cookie(default=None)):
    """Whether the packaged model is ready to download (it takes a few seconds after the last round)."""
    require(attune_session)
    _valid(session)
    if MOCK:
        return {"ready": False, "mock": True}
    if not (FILES_URL and MODAL_KEY):
        return {"ready": False, "configured": False}
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("GET", f"{FILES_URL}/adapter/{session}", params={"key": MODAL_KEY}) as r:
            size = r.headers.get("content-length")
            return {"ready": r.status_code == 200, "bytes": int(size) if size and r.status_code == 200 else None}


@app.get("/api/session/{session}/model")
async def model_download(session: str, attune_session: str | None = Cookie(default=None)):
    require(attune_session)
    _valid(session)
    if MOCK or not (FILES_URL and MODAL_KEY):
        raise HTTPException(status_code=404, detail="Downloads are not available in this mode")
    client = httpx.AsyncClient(timeout=None)
    r = await client.send(client.build_request("GET", f"{FILES_URL}/adapter/{session}", params={"key": MODAL_KEY}),
                          stream=True)
    if r.status_code != 200:
        await r.aclose()
        await client.aclose()
        raise HTTPException(status_code=404, detail="The model is not ready yet, or is older than a day")

    async def body():
        try:
            async for chunk in r.aiter_bytes(1 << 16):
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    headers = {"Content-Disposition": 'attachment; filename="attune-model.zip"'}
    if r.headers.get("content-length"):
        headers["Content-Length"] = r.headers["content-length"]
    return StreamingResponse(body(), media_type="application/zip", headers=headers)


# ---- the built front end (production); in development Vite serves it and proxies /api here ----

DIST = Path(__file__).parent.parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = DIST / path
        return FileResponse(target if path and target.is_file() else DIST / "index.html")
