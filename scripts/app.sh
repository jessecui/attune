#!/bin/zsh
# Run the Attune app locally: backend on :8787, front end on :5180 (open that one).
# The ports are Attune's own, so it can run beside other projects on 8000 and 5173.
#   scripts/app.sh          uses the GPU service on Modal (costs GPU time while a session runs)
#   scripts/app.sh mock     replays a recorded session: no GPU, no cost, for working on the page
cd "$(dirname "$0")/.."
# Settings come from app/backend/.env (not in git): CREATOR_NAME, the name the access prompt asks for, and for the
# real service the two Modal addresses and the Modal proxy token (see app/backend/.env.example).
if [ -f app/backend/.env ]; then set -a; source app/backend/.env; set +a; fi
if [ -z "$CREATOR_NAME" ]; then echo "Set CREATOR_NAME in app/backend/.env (see app/backend/.env.example)."; exit 1; fi
if [ "$1" = "mock" ]; then
  export ATTUNE_MOCK=1
else
  export ATTUNE_MODAL_KEY="$(cat .live_key)"
  for v in ATTUNE_MODAL_URL ATTUNE_FILES_URL ATTUNE_MODAL_PROXY_ID ATTUNE_MODAL_PROXY_SECRET; do
    if [ -z "${(P)v}" ]; then echo "Set $v in app/backend/.env (see app/backend/.env.example), or run: scripts/app.sh mock"; exit 1; fi
  done
fi
[ -d app/frontend/node_modules ] || (cd app/frontend && npm install)
export ATTUNE_BACKEND_PORT="${ATTUNE_BACKEND_PORT:-8787}"
if lsof -nP -iTCP:$ATTUNE_BACKEND_PORT -sTCP:LISTEN >/dev/null || lsof -nP -iTCP:5180 -sTCP:LISTEN >/dev/null; then
  echo "Port $ATTUNE_BACKEND_PORT or 5180 is already in use. Stop whatever is using it, or the earlier copy of this script."; exit 1
fi
# --reload: the backend picks up code changes itself, as the page already does
(cd app/backend && ../../.venv/bin/uvicorn main:app --reload --no-proxy-headers --timeout-graceful-shutdown 2 --port $ATTUNE_BACKEND_PORT) &
backend=$!
# On exit, stop the backend and wait for it, so its last log lines print before the prompt returns.
# Ctrl-C would otherwise end this script at once, without the exit step, so it is caught and turned into an exit.
trap "kill $backend 2>/dev/null; wait $backend 2>/dev/null" EXIT
trap "exit 130" INT TERM
echo "Open http://localhost:5180 (the access prompt asks for the creator's name)"
cd app/frontend && npm run dev
