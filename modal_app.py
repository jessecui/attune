"""The Attune benchmark as a deployed Modal app: one GPU container per run.

Runs are launched detached and write their results to a Modal volume, so nothing
on the laptop has to stay open while they run.

    modal deploy modal_app.py                 # once, and again after code changes
    python -m bench.remote launch main        # returns in seconds
    python -m bench.remote status main        # what is done; downloads finished results
"""
import modal

app = modal.App("attune-bench")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.6", "transformers>=4.51", "peft>=0.15", "accelerate", "google-genai>=1.56.0")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("bench")
    .add_local_dir("data", remote_path="/root/data")
)
hf_cache = modal.Volume.from_name("attune-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("attune-results", create_if_missing=True)
# The judge model's key (GEMINI_API_KEY): modal secret create attune-judge GEMINI_API_KEY=...
judge_key = modal.Secret.from_name("attune-judge")
COMMON = dict(image=image, volumes={"/cache": hf_cache, "/results": results}, secrets=[judge_key],
              timeout=30 * 60, max_containers=12)


def _run(job: dict) -> str:
    """Run one benchmark job and save its log to /results/<grid>/<name>.json."""
    import json
    import os
    import subprocess
    import time
    from pathlib import Path

    os.chdir("/root")
    if job.get("module") == "check_menu":
        from bench.check_menu import main
    else:
        from bench.run import main

    out = Path("/results") / job["grid"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{job['name']}.started").write_text(str(time.time()))
    results.commit()

    t0 = time.time()
    extra = [] if job.get("module") else ["--out", f"/tmp/{job['name']}.json"]
    log = main(job["argv"] + extra)
    log["name"], log["wall_seconds"] = job["name"], time.time() - t0
    log["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip()
    (out / f"{job['name']}.json").write_text(json.dumps(log, indent=1))
    results.commit()
    hf_cache.commit()  # keep downloaded weights for the next container
    return job["name"]


# L40S is the default: in the timing comparison it was both the fastest and the cheapest per run.
@app.function(gpu="L40S", **COMMON)
def run_one(job: dict) -> str:
    return _run(job)


# The same job on other GPU types, for timing comparisons (a job picks one with "fn").
@app.function(gpu="A10", **COMMON)
def run_one_a10(job: dict) -> str:
    return _run(job)


@app.function(gpu="L40S", **COMMON)
def run_one_l40s(job: dict) -> str:
    return _run(job)


@app.function(gpu="A100-40GB", **COMMON)
def run_one_a100(job: dict) -> str:
    return _run(job)


# ---- a check that a downloaded model package runs as its README says --------------------------------
adapters = modal.Volume.from_name("attune-adapters", create_if_missing=True)


@app.function(gpu="L40S", image=image, volumes={"/cache": hf_cache, "/adapters": adapters}, timeout=20 * 60)
def check_package(session: str = "", question: str = "How do I ask for a raise?", example: str = "") -> dict:
    """Unzip a packaged adapter (the newest, unless a session id is given) and run its example.py.

        modal run modal_app.py::check_package
    """
    import json
    import subprocess
    import zipfile
    from pathlib import Path

    zips = sorted(Path("/adapters").glob("*.zip"), key=lambda z: z.stat().st_mtime)
    target = Path("/adapters") / f"{session}.zip" if session else (zips[-1] if zips else None)
    if target is None or not target.exists():
        return {"ok": False, "error": "no packaged adapter found"}
    zipfile.ZipFile(target).extractall("/tmp/pkg")
    pkg = Path("/tmp/pkg/attune-model")
    if example:   # test a newer example.py against an already packaged adapter
        (pkg / "example.py").write_text(example)
    run = subprocess.run(["python", "example.py", question], cwd=pkg, capture_output=True, text=True)
    meta = json.loads((pkg / "attune.json").read_text())
    return {"ok": run.returncode == 0, "package": target.name, "files": sorted(f.name for f in pkg.iterdir()),
            "speaking_as": meta.get("speaking_as"), "style_instructions": meta["style_instructions"],
            "answer": run.stdout.strip()[-1500:], "error": run.stderr.strip()[-1500:] if run.returncode else ""}


@app.local_entrypoint()
def package(session: str = "", question: str = "How do I ask for a raise?"):
    import json
    from modal_live import PACKAGE_EXAMPLE   # the script as it will be packaged from now on
    print(json.dumps(check_package.remote(session, question, PACKAGE_EXAMPLE), indent=1))

