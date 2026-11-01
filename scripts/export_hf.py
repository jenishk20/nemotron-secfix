"""Build the Hugging Face upload folders (nothing is uploaded here).

hf_export/cweval-tasks/        CWEval's 119 tasks as JSONL (Apache-2.0, credited) + our per-model results
hf_export/secure-code-synth/   our verified synthetic tasks per language + the SFT files we trained on

Usage: .venv/bin/python scripts/export_hf.py
"""
import glob, json, os, re, shutil, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "datagen"))
from build_sft import cweval_prompts, leaks  # noqa: E402
from catalog import CWES  # noqa: E402

OUT = os.path.join(ROOT, "hf_export")
BENCH = os.path.join(ROOT, "third_party", "CWEval", "benchmark")
LANG = {"py": "python", "c": "c", "cpp": "cpp", "js": "javascript", "go": "go"}
# runs in outputs/ -> label shown in the results file (all n=5, temperature 0.8, strict scoring)
RUNS = {
    "base_wandb_all_n5": "Nemotron 3.5 Super VL 120B (base)",
    "lightning_wandb_nothink_n5": "Nemotron 3.5 Lightning 30B-A3B (base, thinking off)",
    "lightning_wandb_think_n5": "Nemotron 3.5 Lightning 30B-A3B (base, thinking on)",
    "gemini35flash_all_n5": "Gemini 3.5 Flash",
    "glimmer30b_or_all_n5": "Muse Glimmer 30B (base)",
    "sonnet55_or_py_n5": "Claude Sonnet 5.5 (Python tasks only)",
}


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {os.path.relpath(path, ROOT)}: {len(rows)} rows")


def cweval_tasks():
    rows = []
    for task in sorted(glob.glob(os.path.join(BENCH, "*", "*", "*_task.*"))):
        rel = os.path.relpath(task, BENCH)                       # core/py/cwe_022_0_task.py
        tid = rel.split("_task")[0]                              # core/py/cwe_022_0
        src = open(task).read()
        sol_line = next(l for l in src.splitlines() if "BEGIN SOLUTION" in l)
        prompt = src.split("BEGIN PROMPT")[-1].split(sol_line)[0].strip()  # exactly what CWEval shows the model
        ext = os.path.splitext(task)[1][1:]
        test = task.replace("_task.", "_test.").rsplit(".", 1)[0] + ".py"
        unsafe = task.replace("_task.", "_unsafe.")
        rows.append({
            "task_id": tid, "suite": rel.split("/")[0], "language": LANG[ext],
            "cwe": int(re.search(r"cwe_(\d+)", rel).group(1)),
            "prompt": prompt, "reference_solution": src,
            "insecure_reference": open(unsafe).read() if os.path.exists(unsafe) else None,
            "tests": open(test).read() if os.path.exists(test) else None,
        })
    return rows


def cweval_results():
    rows = []
    for run, label in RUNS.items():
        path = os.path.join(ROOT, "outputs", run, "summary.json")
        if not os.path.exists(path):
            continue
        for tid, v in json.load(open(path))["per_task"].items():
            if "/" not in tid:  # Python-only runs scored before task ids carried the language folder
                tid = f"core/py/{tid}"
            rows.append({"task_id": tid, "model": label, "run": run, "samples": v["n"],
                         "works": v["func"], "works_and_secure": v["func_sec"]})
    return rows


def synth_tasks():
    bench, recs = cweval_prompts(), {}
    files = [f for f in sorted(glob.glob(os.path.join(ROOT, "data", "synth", "*.jsonl")))
             if not f.endswith(".verify.jsonl") and "regress" not in f and "rl_work" not in f]
    for f in files:
        vf = f.replace(".jsonl", ".verify.jsonl")
        if not os.path.exists(vf):
            continue
        ver = {r["id"]: r for r in map(json.loads, open(vf)) if r.get("ok")}
        rnd = 2 if f.endswith(".r2.jsonl") else 1 if f.endswith(".r1.jsonl") else 0
        for r in map(json.loads, open(f)):
            if r["id"] not in ver or r["id"] in recs or leaks(r["stub"], bench):
                continue
            lang = r.get("lang", "py")
            recs[r["id"]] = {
                "task_id": r["id"], "language": LANG[lang], "cwe": r["cwe"], "cwe_description": CWES[lang][r["cwe"]],
                "domain": r["domain"], "prompt": r["stub"], "secure_solution": r["secure"],
                "insecure_solution": r["naive"], "entrypoint": r.get("entrypoint"), "tests": r["tests"],
                "functionality_tests": ver[r["id"]]["n_func"], "security_tests": ver[r["id"]]["n_sec"],
                "repair_rounds": rnd, "teacher": r["teacher"],
            }
    by_lang = {}
    for r in recs.values():
        by_lang.setdefault(r["language"], []).append(r)
    return by_lang


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    print("cweval-tasks/")
    d = os.path.join(OUT, "cweval-tasks")
    write_jsonl(os.path.join(d, "data", "tasks.jsonl"), cweval_tasks())
    write_jsonl(os.path.join(d, "results", "results.jsonl"), cweval_results())
    shutil.copy(os.path.join(ROOT, "third_party", "CWEval", "LICENSE"), os.path.join(d, "LICENSE"))
    shutil.copy(os.path.join(ROOT, "hf_cards", "cweval-tasks.md"), os.path.join(d, "README.md"))

    print("secure-code-synth/")
    d = os.path.join(OUT, "secure-code-synth")
    for lang, rows in sorted(synth_tasks().items()):
        write_jsonl(os.path.join(d, "tasks", f"{lang}.jsonl"), sorted(rows, key=lambda r: r["task_id"]))
    for name in ("secpy_train.jsonl", "secpy_val.jsonl"):
        src = os.path.join(ROOT, "data", name)
        dst = os.path.join(d, "sft", name.replace("secpy_", "python_v1_"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        print(f"  {os.path.relpath(dst, ROOT)}: {sum(1 for _ in open(src))} rows")
    shutil.copy(os.path.join(ROOT, "hf_cards", "secure-code-synth.md"), os.path.join(d, "README.md"))


if __name__ == "__main__":
    main()
