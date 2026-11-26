#!/bin/zsh
# Run the Attune app locally: backend on :8787, front end on :5180 (open that one).
# The ports are Attune's own, so it can run beside other projects on 8000 and 5173.
#   scripts/app.sh          uses the GPU service on Modal (costs GPU time while a session runs)
#   scripts/app.sh mock     replays a recorded session: no GPU, no cost, for working on the page
cd "$(dirname "$0")/.."
export ATTUNE_PASSWORD="${ATTUNE_PASSWORD:-attune}"
if [ "$1" = "mock" ]; then
  export ATTUNE_MOCK=1
else
  export ATTUNE_MODAL_KEY="$(cat .live_key)"
  export ATTUNE_MODAL_URL="${ATTUNE_MODAL_URL:-https://jcui-projects--attune-live-live-web.us-east.modal.run}"
  export ATTUNE_FILES_URL="${ATTUNE_FILES_URL:-https://jcui-projects--attune-live-files.us-east.modal.run}"
fi
[ -d app/frontend/node_modules ] || (cd app/frontend && npm install)
export ATTUNE_BACKEND_PORT="${ATTUNE_BACKEND_PORT:-8787}"
if lsof -nP -iTCP:$ATTUNE_BACKEND_PORT -sTCP:LISTEN >/dev/null || lsof -nP -iTCP:5180 -sTCP:LISTEN >/dev/null; then
  echo "Port $ATTUNE_BACKEND_PORT or 5180 is already in use. Stop whatever is using it, or the earlier copy of this script."; exit 1
fi
# --reload: the backend picks up code changes itself, as the page already does
(cd app/backend && ../../.venv/bin/uvicorn main:app --reload --port $ATTUNE_BACKEND_PORT) &
backend=$!
trap "kill $backend 2>/dev/null" EXIT
echo "Password: $ATTUNE_PASSWORD   Open http://localhost:5180"
cd app/frontend && npm run dev
