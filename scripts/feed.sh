#!/bin/zsh
# Feed a grid to Modal a few jobs at a time: keep at most N queued or running (default 10).
cd "$(dirname "$0")/.."
grid=$1; cap=${2:-10}
for i in $(seq 1 120); do
  s=$(.venv/bin/python -m bench.remote status $grid 2>&1 | grep -v Warning | tail -1)
  total=$(.venv/bin/python -c "import bench.remote as B; print(len(B.GRIDS['$grid']))" 2>/dev/null | tail -1)
  case "$s" in *"$grid: $total done"*) break;; esac
  active=$(.venv/bin/python -c "
import modal
s = modal.Function.from_name('attune-bench', 'run_one').get_current_stats()
print(s.backlog + s.num_running_inputs)" 2>/dev/null | tail -1)
  if [ -n "$active" ] && [ "$active" -lt $cap ]; then
    echo "$(date +%H:%M) $(.venv/bin/python -m bench.remote launch $grid --limit $((cap-active)) 2>&1 | grep -v Warning | tail -1)"
  fi
  sleep 45
done
date; .venv/bin/python -m bench.remote status $grid 2>&1 | grep -v Warning | grep -E "done,|FAILED" | cut -c1-120
