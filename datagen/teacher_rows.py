"""SFT rows from the teacher's own verified secure solutions, for a given list of synthetic task ids.

Used for CyberNemo v4: on synthetic tasks where Nemotron 120B never produced a passing answer, the
training answer is Nemotron 3 Ultra's secure reference solution, which passed all functionality and
security tests in the offline sandbox when the task was verified. CWEval leakage filter applied.
Usage: .venv/bin/python datagen/teacher_rows.py <ids_file> <n> <out.jsonl>
"""
import glob, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "datagen")); sys.path.insert(0, os.path.join(ROOT, "third_party", "CWEval"))
from build_sft import FENCE, cweval_prompts, leaks  # noqa: E402
from cweval.ppt import DirectPrompt  # noqa: E402

ids_file, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ids = [l.strip() for l in open(ids_file) if l.strip()][:n]
want, recs = set(ids), {}
for f in sorted(glob.glob(os.path.join(ROOT, "data", "synth", "*.jsonl"))):
    vf = f.replace(".jsonl", ".verify.jsonl")
    if f.endswith(".verify.jsonl") or "/sd_" in f or "regress" in f or not os.path.exists(vf):
        continue
    ok = {r["id"] for r in map(json.loads, open(vf)) if r.get("ok")}
    for r in map(json.loads, open(f)):
        if r["id"] in want and r["id"] in ok and r["id"] not in recs:
            recs[r["id"]] = r
bench, rows = cweval_prompts(), []
for tid in ids:
    r = recs.get(tid)
    if not r or leaks(r["stub"], bench):
        continue
    lang = r.get("lang", "py")
    user = DirectPrompt.PPT.format(lang=lang, lang_instr=DirectPrompt.LANG_INSTR[lang], code_prompt=r["stub"].rstrip("\n"))
    rows.append({"messages": [{"role": "user", "content": user},
                              {"role": "assistant", "content": f"```{FENCE[lang]}\n{r['secure'].rstrip()}\n```"}]})
with open(out, "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
print(f"{len(rows)} teacher rows from {len(ids)} requested tasks -> {out}")
