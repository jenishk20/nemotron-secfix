"""Rebuild a run's summary.json from only the answers that actually exist.

The review harness was stopped by its billing cap after 869 of 1,000 answers. CWEval's scorer
still visits every task in each generated_* folder and marks a missing answer as "no code",
which would count un-reviewed samples as failures. This keeps only samples whose *_raw.* file
exists, so every task is averaged over the answers it really has (all 100 tasks are covered).
Usage: python scripts/summarize_existing.py <run> [--out results/<run>/summary.json]
"""
import argparse, collections, glob, json, os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--out")
a = ap.parse_args()
run_dir = os.path.join(ROOT, "outputs", a.run)
per_task, kept = collections.defaultdict(list), 0
for g in sorted(glob.glob(os.path.join(run_dir, "generated_*", "res.json"))):
    gdir = os.path.dirname(g)
    for k, v in json.load(open(g)).items():
        t = re.split(r"generated_\d+/", k)[-1].split("_test")[0]
        if glob.glob(os.path.join(gdir, f"{t}_raw.*")):
            per_task[t].append(v); kept += 1
summary = {"tasks": len(per_task), "samples": kept, "note": "only answers that exist (see scripts/summarize_existing.py)",
           "per_task": {t: {"n": len(vs), "func": sum(v["functional"] for v in vs),
                            "func_sec": sum(v["functional"] and v["secure"] for v in vs)} for t, vs in sorted(per_task.items())}}
out = a.out or os.path.join(ROOT, "results", a.run, "summary.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(summary, open(out, "w"), indent=2)
print(f"{a.run}: {kept} answers over {len(per_task)} tasks -> {os.path.relpath(out, ROOT)}")
