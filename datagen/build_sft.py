"""Verified candidates -> Fireworks chat JSONL, after a leakage filter vs CWEval.

The user turn is exactly CWEval's `direct` prompt (what the model sees at eval
time); the assistant turn is the verified secure module in one code block.
Usage: .venv/bin/python datagen/build_sft.py data/synth/py_pilot.jsonl [more.jsonl ...] \
         --out data/train_sec.jsonl [--holdout-cwes 117,643] [--per-lang 250] [--val 0]
--per-lang N keeps N examples per language, spread evenly over CWEs (round-robin, seeded);
--lang-cap py=250 overrides that cap for the named languages;
--val M also writes M held-out rows per language to <out>.val.jsonl.
"""
import argparse, difflib, glob, json, os, random, re, sys
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


def stub_name(stub):
    """Function name from a stub: Python `def f(`, or a C/C++/JS signature line ending in `{`."""
    m = re.search(r"def\s+(\w+)\s*\(", stub) or re.search(r"(\w+)\s*\([^()]*\)\s*(?:const\s*)?\{\s*$", stub.rstrip())
    return m.group(1)


def leaks(stub, bench):
    name = stub_name(stub)
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
    ap.add_argument("--per-lang", type=int, default=0)
    ap.add_argument("--lang-cap", default="")  # e.g. py=250,c=100
    ap.add_argument("--val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    holdout = {int(x) for x in a.holdout_cwes.split(",") if x}
    bench = cweval_prompts()
    rows, why, kept_cwe, seen = [], Counter(), Counter(), set()
    for inp in a.inputs:
        ver = {r["id"]: r for r in map(json.loads, open(inp.replace(".jsonl", ".verify.jsonl")))}
        for rec in map(json.loads, open(inp)):
            if not ver.get(rec["id"], {}).get("ok"):
                why["not_verified"] += 1; continue
            if rec["cwe"] in holdout:
                why["holdout"] += 1; continue
            if rec["id"] in seen:  # an id verified in more than one repair round
                why["duplicate"] += 1; continue
            seen.add(rec["id"])
            if (l := leaks(rec["stub"], bench)):
                why["leak"] += 1; print("LEAK", rec["id"], l); continue
            user = DirectPrompt.PPT.format(lang=rec["lang"], lang_instr=DirectPrompt.LANG_INSTR[rec["lang"]],
                                           code_prompt=rec["stub"].rstrip("\n"))
            asst = f"```{FENCE[rec['lang']]}\n{rec['secure'].rstrip()}\n```"
            rows.append({"lang": rec["lang"], "cwe": rec["cwe"],
                         "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": asst}]})

    train, val = rows, []
    caps = {k: int(v) for k, v in (x.split("=") for x in a.lang_cap.split(",") if x)}
    if a.per_lang or a.val or caps:
        rng, train = random.Random(a.seed), []
        for lang in sorted({r["lang"] for r in rows}):
            by_cwe = {}
            for r in rows:
                if r["lang"] == lang:
                    by_cwe.setdefault(r["cwe"], []).append(r)
            for lst in by_cwe.values():
                rng.shuffle(lst)
            picked, want = [], caps.get(lang, a.per_lang or 10**9) + a.val
            while len(picked) < want and any(by_cwe.values()):  # round-robin over CWEs
                for c in sorted(by_cwe):
                    if by_cwe[c] and len(picked) < want:
                        picked.append(by_cwe[c].pop())
            rng.shuffle(picked)
            val += picked[:a.val]
            train += picked[a.val:]
            print(f"{lang}: {len(picked) - a.val} train + {min(a.val, len(picked))} val (of {sum(r['lang'] == lang for r in rows)} available)")
        rng.shuffle(train)
    for path, part in [(a.out, train)] + ([(a.out.replace(".jsonl", ".val.jsonl"), val)] if a.val else []):
        with open(path, "w") as f:
            for r in part:
                f.write(json.dumps({"messages": r["messages"]}) + "\n")
        print(f"wrote {len(part)} rows to {path}")
    for r in train:
        kept_cwe[(r["lang"], r["cwe"])] += 1
    print(f"dropped {dict(why)}")
    print("per language/CWE:", dict(sorted(kept_cwe.items())))


if __name__ == "__main__":
    main()
