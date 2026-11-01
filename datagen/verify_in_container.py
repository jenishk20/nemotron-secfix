"""Runs INSIDE the CWEval container (offline). Keeps a candidate only if:
  1. SECURE passes every test (>=4 functionality, >=2 security),
  2. NAIVE passes the functionality tests,
  3. NAIVE fails at least one security test (so the tests really check security).

C/C++/JS programs are assembled the way CWEval scores them: CWEval's header block (C/C++)
+ the answer + "// BEGIN ENTRYPOINT" + the entrypoint, compiled with CWEval's libraries
(plus AddressSanitizer, so memory errors crash). Tests reach the program via harness.run_impl.
Usage: python verify_in_container.py <in.jsonl> <out.jsonl> [--workers 6]
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

PYTEST_INI = "[pytest]\nmarkers =\n    functionality\n    security\n"
HARNESS = """import os, subprocess
CMD = os.environ["IMPL_CMD"].split()


def run_impl(*args, input=None, timeout=10):
    p = subprocess.run(CMD + [str(a) for a in args], capture_output=True, text=True, input=input, timeout=timeout)
    return p.returncode, p.stdout, p.stderr
"""
EXT = {"py": "py", "c": "c", "cpp": "cpp", "js": "js"}
LIBS = "-lsqlite3 -ljwt -lssl -lcrypto -larchive $(xml2-config --cflags --libs) -fsanitize=address"


def build(d, rec, which):
    """Write (and compile) the program for rec[which]; returns (ok, log)."""
    lang = rec.get("lang", "py")
    if lang == "py":
        open(os.path.join(d, "impl.py"), "w").write(rec[which])
        return True, ""
    from cweval.commons import complete_code  # CWEval's header block for C/C++ (no-op for JS)
    src = complete_code(f"{rec[which]}\n\n// BEGIN ENTRYPOINT\n{rec['entrypoint']}\n", lang)
    path = os.path.join(d, f"impl.{EXT[lang]}")
    open(path, "w").write(src)
    if lang == "js":
        return True, ""
    cc = "gcc" if lang == "c" else "g++ -std=c++17"
    p = subprocess.run(f"{cc} {path} -o {os.path.join(d, 'impl')} {LIBS}", shell=True,
                       capture_output=True, text=True, timeout=120)
    return p.returncode == 0, (p.stdout + p.stderr)[-1500:]


def impl_env(d, lang):
    cmd = {"c": os.path.join(d, "impl"), "cpp": os.path.join(d, "impl"), "js": f"node {os.path.join(d, 'impl.js')}"}
    return dict(os.environ, IMPL_CMD=cmd.get(lang, ""), ASAN_OPTIONS="detect_leaks=0")


def func_name(rec):
    """Function name from the stub, and whether the answer/tests refer to it."""
    lang, stub = rec.get("lang", "py"), rec["stub"]
    if lang == "py":
        fn = re.search(r"def\s+(\w+)\s*\(", stub)
        return fn and f"def {fn.group(1)}" in rec["secure"] and f"import {fn.group(1)}" in rec["tests"]
    if not stub.rstrip().endswith("{") or re.search(r"#include|require\(", stub):
        return False
    fn = re.search(r"(\w+)\s*\([^()]*\)\s*(?:const\s*)?\{\s*$", stub.rstrip())
    return bool(fn) and fn.group(1) in rec["secure"] and fn.group(1) in rec["entrypoint"]


def pytest(d, mark=None, timeout=60, env=None):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_task.py"]
    if mark:
        cmd += ["-m", mark]
    try:
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout, env=env)
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
    if not func_name(rec):
        return dict(out, ok=False, why="name_mismatch")
    lang = rec.get("lang", "py")
    d = tempfile.mkdtemp(prefix="v_")
    env = impl_env(d, lang)
    try:
        open(os.path.join(d, "pytest.ini"), "w").write(PYTEST_INI)
        open(os.path.join(d, "harness.py"), "w").write(HARNESS)
        open(os.path.join(d, "test_task.py"), "w").write(rec["tests"])
        ok, log = build(d, rec, "secure")
        if not ok:
            return dict(out, ok=False, why="secure_compile", log=log)
        n_func = pytest(d, "functionality", env=env)
        n_sec = pytest(d, "security", env=env)
        if n_func[0] != 0 or n_sec[0] != 0:
            return dict(out, ok=False, why="secure_fails", log=(n_func[3] if n_func[0] != 0 else n_sec[3]))
        if n_func[1] < 4 or n_sec[1] < 2:
            return dict(out, ok=False, why=f"too_few_tests f={n_func[1]} s={n_sec[1]}")
        for f in ("__pycache__",):
            shutil.rmtree(os.path.join(d, f), ignore_errors=True)
        ok, log = build(d, rec, "naive")
        if not ok:
            return dict(out, ok=False, why="naive_compile", log=log)
        nf = pytest(d, "functionality", env=env)
        if nf[0] != 0:
            return dict(out, ok=False, why="naive_not_functional", log=nf[3])
        ns = pytest(d, "security", env=env)
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
