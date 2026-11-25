#!/bin/zsh
# Run the Attune app locally: backend on :8000, front end on :5173 (open that one).
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
(cd app/backend && ../../.venv/bin/uvicorn main:app --port 8000) &
backend=$!
trap "kill $backend 2>/dev/null" EXIT
echo "Password: $ATTUNE_PASSWORD   Open http://localhost:5173"
cd app/frontend && npm run dev
