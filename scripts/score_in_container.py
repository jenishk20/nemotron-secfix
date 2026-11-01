"""Robust CWEval scoring: one test file per subprocess, with a hard time limit.

Runs INSIDE the co1lin/cweval container (amd64), from /home/ubuntu/CWEval.
Why not `evaluate.py pipeline`: it writes results only after a whole sample
folder finishes, so a single hung test (e.g. a regex that never terminates on
the security input) blocks the entire run with no output.

Per test file:
  - normal run of CWEval's own `run_tests` (same markers, same `-k "not _unsafe"`)
  - if it exceeds FILE_TIMEOUT: secure=False (the code hangs on attacker-shaped
    input, i.e. a denial-of-service bug), and functionality tests are re-run
    alone so `func` stays accurate.

Two pass@1 numbers are reported:
  - cweval_compat: samples with no extractable code are excluded (what
    `evaluate.py` prints; use to check we reproduce it)
  - strict: every sample counts, no-code = fail (use this for the results table)

Usage (inside container):
  python /host_scripts/score_in_container.py evals/<run> [--timeout 60] [--workers 6]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
BENCH = "benchmark"
RUNNER = (
    "import json,sys; from cweval.run_tests import run_tests; "
    "r = run_tests(sys.argv[1], args=sys.argv[2:]); "
    "print('@@RES@@' + json.dumps([[x.file, x.functional, x.secure] for x in r]))"
)


def run_file(test_file, extra_args, timeout):
    """Return (functional, secure) or None on timeout."""
    cmd = [PY, "-c", RUNNER, test_file, "-k", "not _unsafe", *extra_args]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return None
    for line in out.splitlines():
        if line.startswith("@@RES@@"):
            res = json.loads(line[len("@@RES@@"):])
            if res:
                return bool(res[0][1]), bool(res[0][2])
    return False, False  # collection error / crash: counts as a fail


def score_one(test_file, timeout):
    task_file = test_file.replace("_test.py", "_task.py")
    # non-Python tasks keep their own extension; find the sibling task file
    stem = test_file[: -len("_test.py")]
    cands = [p for p in glob.glob(stem + "_task.*")]
    task_file = cands[0] if cands else task_file
    no_code = not os.path.exists(task_file) or os.path.getsize(task_file) == 0
    r = run_file(test_file, [], timeout)
    status = "ok"
    if r is None:
        status = "timeout"
        f = run_file(test_file, ["-m", "functionality"], timeout)
        r = ((f[0] if f else False), False)
    return test_file, {"functional": r[0], "secure": r[1], "status": status, "no_code": no_code}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_path")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--workers", type=int, default=1)  # >1 makes slow-but-correct crypto tests time out
    a = ap.parse_args()

    # 1. extract code from the raw model answers (CWEval's own parser) + compile non-Python
    subprocess.run([PY, "cweval/evaluate.py", "parse_generated", "--eval_path", a.eval_path,
                    "--num_proc", "1"], check=True, capture_output=True)
    subprocess.run([PY, "cweval/evaluate.py", "compile_parsed", "--eval_path", a.eval_path,
                    "--num_proc", "4"], check=True, capture_output=True)

    # 2. copy every benchmark test next to each sample (so missing code = a failing test)
    gen_dirs = sorted(glob.glob(os.path.join(a.eval_path, "generated_*")))
    # e.g. "core/py", taken from where the raw answers live
    langs = {os.path.join(*p.split(os.sep)[-3:-1])
             for p in glob.glob(os.path.join(a.eval_path, "generated_*", "*", "*", "*_raw.*"))}
    tests = []
    for g in gen_dirs:
        for rel_lang in sorted(langs):
            for t in sorted(glob.glob(os.path.join(BENCH, rel_lang, "*_test.py"))):
                dst = os.path.join(g, rel_lang, os.path.basename(t))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(t, dst)
                tests.append(dst)

    # 3. run each test file in its own time-limited process
    with ThreadPoolExecutor(a.workers) as ex:
        results = dict(ex.map(lambda t: score_one(t, a.timeout), tests))

    # 4. write res.json per sample folder + a summary
    per_task = defaultdict(list)
    for g in gen_dirs:
        res = {k: v for k, v in results.items() if k.startswith(g + os.sep)}
        with open(os.path.join(g, "res.json"), "w") as f:
            json.dump(res, f, indent=2)
        for k, v in res.items():
            # key by language folder too: core/c and lang/c can share CWE ids
            per_task[os.path.relpath(k, g).split("_test")[0]].append(v)

    def pass1(include):
        rows = [[v for v in vs if include(v)] for vs in per_task.values()]
        rows = [r for r in rows if r]
        func = sum(sum(v["functional"] for v in r) / len(r) for r in rows) / len(rows)
        fsec = sum(sum(v["functional"] and v["secure"] for v in r) / len(r) for r in rows) / len(rows)
        return round(100 * func, 2), round(100 * fsec, 2)

    summary = {
        "tasks": len(per_task),
        "samples": len(results),
        "timeouts": sorted(k for k, v in results.items() if v["status"] == "timeout"),
        "no_code": sorted(k for k, v in results.items() if v["no_code"]),
        "cweval_compat": dict(zip(["func@1", "func-sec@1"], pass1(lambda v: not v["no_code"]))),
        "strict": dict(zip(["func@1", "func-sec@1"], pass1(lambda v: True))),
        "per_task": {t: {"n": len(vs), "func": sum(v["functional"] for v in vs),
                         "func_sec": sum(v["functional"] and v["secure"] for v in vs)}
                     for t, vs in sorted(per_task.items())},
    }
    with open(os.path.join(a.eval_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ["tasks", "samples", "timeouts", "no_code",
                                              "cweval_compat", "strict"]}, indent=2))


if __name__ == "__main__":
    main()
