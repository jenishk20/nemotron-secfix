"""Ask a teacher model for new secure-coding tasks (CWEval style).

Each candidate = task stub (what the student sees) + secure solution +
naive solution + pytest file. Nothing here is trusted until
verify_in_container.py has executed it.

Usage: .venv/bin/python datagen/teacher.py --n 20 --out data/synth/py_pilot.jsonl
       .venv/bin/python datagen/teacher.py --repair data/synth/py_pilot.jsonl --out data/synth/py_pilot.r1.jsonl
Resumable: ids already in --out are skipped.
"""
import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from catalog import DOMAINS, PY_CWES, PY_LIBS

ROOT = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(os.path.join(ROOT, ".env"))
TEACHER = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"

PROMPT = """You write training tasks for teaching a code model to write SECURE code. Create ONE new, original Python coding task.

Weakness to exercise: CWE-{cwe}: {cwe_desc}
Application area: {domain}
Variation seed: {seed} (use it to pick an unusual but realistic function; avoid the most obvious textbook example)

Requirements:
- A single focused Python function (helpers allowed), 10-40 lines. Prefer a clear, precisely specified behavior over a big feature.
- Allowed libraries: {libs}. Import everything you use; no other third-party packages exist. No network access is available when tests run; never make real network calls (for URL tasks, the function should build/validate URLs or take an injectable fetch callable).
- The docstring must describe the behavior, arguments and return value precisely, like a real spec, and say which inputs come from users. It must NOT name the vulnerability or tell the reader how to defend against it. A careful engineer should still write the secure version.
- The function must always return gracefully as the docstring specifies (e.g. return an empty string / None / False on bad input) unless the docstring says it raises a specific exception.

Write these four sections, each in a single fenced code block, in this exact order and with these exact headers:

### STUB
Imports needed, then the function signature and full docstring, with no body (end right after the docstring).

### SECURE
The complete module: imports + the full secure implementation (same signature and docstring).

### NAIVE
The complete module: imports + a plausible implementation a hurried developer would write. It must work on normal inputs but contain the CWE-{cwe} weakness.

### TESTS
A pytest file. Import the function with `from impl import <function_name>`. Requirements:
- Mark every test with exactly one of `@pytest.mark.functionality` or `@pytest.mark.security`.
- At least 4 functionality tests (normal use, edge cases) and at least 2 security tests (hostile inputs that check the weakness is not present: e.g. assert the traversal file is NOT read, the injected command did NOT run, output IS escaped, the forged token IS rejected).
- The secure version must pass ALL tests; the naive version must pass all functionality tests and FAIL at least one security test.
- Functionality tests may only check behavior the docstring specifies exactly (no guessing at formatting the docstring leaves open).
- Tests must be deterministic, offline, fast (< 5 seconds total), use pytest's tmp_path for any files, and must not depend on exact error messages.
"""

SECTION_RE = re.compile(r"###\s*(STUB|SECURE|NAIVE|TESTS)\s*\n+```[a-zA-Z]*\n(.*?)```", re.S)


def parse(text):
    found = {k.lower(): v.strip() + "\n" for k, v in SECTION_RE.findall(text or "")}
    return found if len(found) == 4 else None


REPAIR = """Your task was executed in a sandbox and did not pass verification.

Verification rules: SECURE must pass all tests; NAIVE must pass all functionality tests and fail at least one security test.
Problem: {why}
Pytest output (tail):
```
{log}
```

Fix whichever section(s) are wrong (stub, implementation, or tests) while keeping the same task, function name and CWE. Do NOT make the security tests weaker than a real defense requires. Reply with all four sections again (### STUB, ### SECURE, ### NAIVE, ### TESTS), each in one fenced code block."""


def call(client, messages, temperature, max_tokens):
    r = client.chat.completions.create(model=TEACHER, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content, r.usage


def answer_text(rec):
    return "\n\n".join(f"### {k.upper()}\n```python\n{rec[k]}```" for k in ("stub", "secure", "naive", "tests"))


def repair_jobs(src):
    """Failed-but-fixable candidates from <src>.jsonl + <src>.verify.jsonl, as repair jobs."""
    ver = {r["id"]: r for r in map(json.loads, open(src.replace(".jsonl", ".verify.jsonl")))}
    jobs = []
    for rec in map(json.loads, open(src)):
        v = ver.get(rec["id"], {})
        if "stub" not in rec or v.get("ok") or v.get("why") in ("unparsed", "name_mismatch"):
            continue
        why = {"secure_fails": "SECURE fails some tests.",
               "naive_not_functional": "NAIVE fails some functionality tests.",
               }.get(v["why"].split(" ")[0], v["why"])
        if v["why"].startswith("naive_passes_security"):
            why = "NAIVE passes every security test, so the security tests do not detect the weakness."
        job = {k: rec[k] for k in ("id", "lang", "cwe", "domain", "seed")}
        job["fix"] = {"prev": rec, "why": why, "log": v.get("log", "(none)")[-1200:]}
        jobs.append(job)
    return jobs


def main(n=20, out="data/synth/py_pilot.jsonl", threads=16, seed=0, temperature=0.8, max_tokens=24000, repair=None):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rng = random.Random(seed)
    cwes = sorted(PY_CWES)
    # Balanced over CWEs, random domain and variation seed per job
    jobs = [{"id": f"py_{seed}_{i:05d}", "lang": "py", "cwe": cwes[i % len(cwes)],
             "domain": rng.choice(DOMAINS), "seed": rng.randrange(10**6)} for i in range(n)]
    if repair:
        jobs = repair_jobs(repair)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    jobs = [j for j in jobs if j["id"] not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done)", flush=True)

    client = OpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=os.environ["WANDB_API_KEY"],
                    project=os.environ.get("WANDB_PROJECT_PATH"), timeout=900, max_retries=2)
    lock = threading.Lock()
    stats = {"ok": 0, "unparsed": 0, "error": 0, "out_tok": 0}
    t0 = time.time()

    def run(job):
        msgs = [{"role": "user", "content": PROMPT.format(cwe=job["cwe"], cwe_desc=PY_CWES[job["cwe"]],
                                                          domain=job["domain"], seed=job["seed"], libs=PY_LIBS)}]
        if "fix" in job:  # repair round: show the previous answer and its sandbox failure
            msgs += [{"role": "assistant", "content": answer_text(job["fix"]["prev"])},
                     {"role": "user", "content": REPAIR.format(why=job["fix"]["why"], log=job["fix"]["log"])}]
        t = time.time()
        text, usage = call(client, msgs, temperature, max_tokens)
        job = {k: v for k, v in job.items() if k != "fix"}
        rec = dict(job, teacher=TEACHER, secs=round(time.time() - t, 1),
                   out_tokens=usage.completion_tokens, in_tokens=usage.prompt_tokens)
        parts = parse(text)
        if parts:
            rec.update(parts)
        else:
            rec["raw"] = text
        return rec

    with ThreadPoolExecutor(threads) as ex, open(out, "a") as f:
        futs = [ex.submit(run, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                stats["error"] += 1
                print(f"ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            with lock:
                stats["ok" if "stub" in rec else "unparsed"] += 1
                stats["out_tok"] += rec["out_tokens"]
                f.write(json.dumps(rec) + "\n"); f.flush()
                k = stats["ok"] + stats["unparsed"]
                if k % 10 == 0 or k == len(jobs):
                    print(f"{k}/{len(jobs)} {stats} {time.time() - t0:.0f}s", flush=True)
    print("final", stats, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/synth/py_pilot.jsonl")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repair", help="candidates .jsonl (already verified) whose failures to send back for a fix")
    main(**vars(ap.parse_args()))
