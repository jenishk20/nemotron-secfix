"""Verified candidates -> Fireworks chat JSONL, after a leakage filter vs CWEval.

The user turn is exactly CWEval's `direct` prompt (what the model sees at eval
time); the assistant turn is the verified secure module in one code block.
Usage: .venv/bin/python datagen/build_sft.py data/synth/py_pilot.jsonl [more.jsonl ...] \
         --out data/train_sec.jsonl [--holdout-cwes 117,643]
"""
import argparse, difflib, glob, json, os, re, sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "third_party", "CWEval"))
from cweval.ppt import DirectPrompt

FENCE = {"py": "python", "js": "javascript", "c": "c", "cpp": "cpp"}


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def cweval_prompts():
    out = []
    for p in glob.glob(os.path.join(ROOT, "third_party/CWEval/benchmark/**/*_task.*"), recursive=True):
        src = open(p).read()
        prompt = re.split(r"BEGIN SOLUTION", src)[0]
        names = set(re.findall(r"(?:def|function|func)\s+(\w+)\s*\(", prompt))
        names |= set(re.findall(r"^[A-Za-z_][\w\s\*:<>,]*?\b(\w+)\s*\([^;]*\)\s*\{", prompt, re.M))  # C/C++ definitions
        out.append((os.path.relpath(p, ROOT), names, norm(prompt)))
    return out


def ngrams(s, n=8):
    w = s.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def leaks(stub, bench):
    name = re.search(r"def\s+(\w+)\s*\(", stub).group(1)
    s = norm(stub); g = ngrams(s)
    for path, names, b in bench:
        if name in names:
            return f"same function name as {path}"
        if g & ngrams(b):
            return f"8-gram overlap with {path}"
        if difflib.SequenceMatcher(None, s, b).quick_ratio() > 0.8 and difflib.SequenceMatcher(None, s, b).ratio() > 0.6:
            return f"similar to {path}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="data/train_sec.jsonl")
    ap.add_argument("--holdout-cwes", default="")
    a = ap.parse_args()
    holdout = {int(x) for x in a.holdout_cwes.split(",") if x}
    bench = cweval_prompts()
    rows, why, kept_cwe = [], Counter(), Counter()
    for inp in a.inputs:
        ver = {r["id"]: r for r in map(json.loads, open(inp.replace(".jsonl", ".verify.jsonl")))}
        for rec in map(json.loads, open(inp)):
            if not ver.get(rec["id"], {}).get("ok"):
                why["not_verified"] += 1; continue
            if rec["cwe"] in holdout:
                why["holdout"] += 1; continue
            if (l := leaks(rec["stub"], bench)):
                why["leak"] += 1; print("LEAK", rec["id"], l); continue
            user = DirectPrompt.PPT.format(lang=rec["lang"], lang_instr=DirectPrompt.LANG_INSTR[rec["lang"]],
                                           code_prompt=rec["stub"].rstrip("\n"))
            asst = f"```{FENCE[rec['lang']]}\n{rec['secure'].rstrip()}\n```"
            rows.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": asst}]})
            kept_cwe[rec["cwe"]] += 1
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {a.out}; dropped {dict(why)}")
    print("per CWE:", dict(sorted(kept_cwe.items())))


if __name__ == "__main__":
    main()
