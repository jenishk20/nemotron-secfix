"""Runs INSIDE the CWEval container (offline). Keeps a candidate only if:
  1. SECURE passes every test (>=4 functionality, >=2 security),
  2. NAIVE passes the functionality tests,
  3. NAIVE fails at least one security test (so the tests really check security).
Usage: python verify_in_container.py <in.jsonl> <out.jsonl> [--workers 6]
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

PYTEST_INI = "[pytest]\nmarkers =\n    functionality\n    security\n"


def pytest(d, mark=None, timeout=60):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_task.py"]
    if mark:
        cmd += ["-m", mark]
    try:
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", 0, 0, ""
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    return p.returncode, passed, failed, (p.stdout + p.stderr)[-1500:]


def verify(rec):
    out = {"id": rec["id"]}
    if "stub" not in rec:
        return dict(out, ok=False, why="unparsed")
    fn = re.search(r"def\s+(\w+)\s*\(", rec["stub"])
    if not fn or f"def {fn.group(1)}" not in rec["secure"] or f"import {fn.group(1)}" not in rec["tests"]:
        return dict(out, ok=False, why="name_mismatch")
    d = tempfile.mkdtemp(prefix="v_")
    try:
        open(os.path.join(d, "pytest.ini"), "w").write(PYTEST_INI)
        open(os.path.join(d, "test_task.py"), "w").write(rec["tests"])
        open(os.path.join(d, "impl.py"), "w").write(rec["secure"])
        n_func = pytest(d, "functionality")
        n_sec = pytest(d, "security")
        if n_func[0] != 0 or n_sec[0] != 0:
            return dict(out, ok=False, why="secure_fails", log=(n_func[3] if n_func[0] != 0 else n_sec[3]))
        if n_func[1] < 4 or n_sec[1] < 2:
            return dict(out, ok=False, why=f"too_few_tests f={n_func[1]} s={n_sec[1]}")
        open(os.path.join(d, "impl.py"), "w").write(rec["naive"])
        for f in ("__pycache__",):
            shutil.rmtree(os.path.join(d, f), ignore_errors=True)
        nf = pytest(d, "functionality")
        if nf[0] != 0:
            return dict(out, ok=False, why="naive_not_functional", log=nf[3])
        ns = pytest(d, "security")
        if ns[0] != 1 or ns[2] < 1:
            return dict(out, ok=False, why=f"naive_passes_security rc={ns[0]}")
        return dict(out, ok=True, why="ok", n_func=n_func[1], n_sec=n_sec[1], naive_sec_failed=ns[2])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.inp)]
    res = []
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(verify, recs):
            res.append(r)
            if len(res) % 50 == 0:
                print(f"verified {len(res)}/{len(recs)}, kept {sum(x['ok'] for x in res)}", flush=True)
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(Counter(r["why"].split(" ")[0] for r in res), f"kept {sum(r['ok'] for r in res)}/{len(res)}")


if __name__ == "__main__":
    main()
