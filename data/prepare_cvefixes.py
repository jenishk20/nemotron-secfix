"""Turn CVEfixes into Fireworks chat-JSONL training pairs: vulnerable code -> fixed code.

CVEfixes (https://zenodo.org/records/13118970) ships as a SQL dump. Load it into
SQLite first:
    # one-time, from the extracted dump:
    #   bzip2 -dk CVEfixes_v1.0.8.sql.bz2
    #   sqlite3 data/CVEfixes.db < CVEfixes_v1.0.8.sql

Schema used below (CVEfixes v1.0.x). COLUMN/TABLE NAMES MAY NEED TWEAKING after you
inspect the actual db with `.tables` and `.schema` — this is the #1 thing to verify
this weekend. Relevant tables:
    file_change(code_before, code_after, programming_language, hash, filename)
    fixes(cve_id, hash)                         # links a commit hash to a CVE
    cwe_classification(cve_id, cwe_id)          # links a CVE to its CWE

Run:  python data/prepare_cvefixes.py
Out:  data/train_secfix.jsonl  (OpenAI chat format, one json per line)
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SYSTEM = (
    "You are a secure-coding assistant. You are given code that contains a security "
    "vulnerability. Rewrite it so it is both functionally correct AND secure. "
    "Return only the corrected code."
)

QUERY = """
SELECT fc.code_before, fc.code_after, fc.programming_language, cc.cwe_id
FROM file_change fc
JOIN fixes f        ON fc.hash = f.hash
JOIN cwe_classification cc ON f.cve_id = cc.cve_id
WHERE fc.code_before IS NOT NULL
  AND fc.code_after  IS NOT NULL
  AND fc.code_before != fc.code_after
  AND fc.programming_language IN ({langs})
"""


def build_pairs(db_path: str):
    langs = ",".join("'%s'" % l for l in config.TRAIN_LANGUAGES)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(QUERY.format(langs=langs)).fetchall()
    con.close()

    for r in rows:
        before, after = r["code_before"].strip(), r["code_after"].strip()
        # Skip trivial / huge diffs that make poor training signal.
        if not before or not after or len(before) > 8000 or len(after) > 8000:
            continue
        cwe = r["cwe_id"] or "unknown weakness"
        user = (
            f"The following {r['programming_language']} code has a security "
            f"vulnerability ({cwe}). Fix it while keeping it functional:\n\n"
            f"```\n{before}\n```"
        )
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": f"```\n{after}\n```"},
            ],
            # kept for dedup/analysis, strip before uploading if Fireworks rejects extra keys:
            "_cwe": cwe,
            "_lang": r["programming_language"],
        }


def main():
    db = config.CVEFIXES_SQLITE
    if not Path(db).exists():
        sys.exit(f"[!] {db} not found. Download CVEfixes from Zenodo and load the SQL dump first.")

    out = Path(config.TRAIN_JSONL)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    seen = set()
    with out.open("w") as fh:
        for rec in build_pairs(db):
            # cheap de-dup on the (before, after) pair
            key = hash((rec["messages"][1]["content"], rec["messages"][2]["content"]))
            if key in seen:
                continue
            seen.add(key)
            fh.write(json.dumps(rec) + "\n")
            n += 1
    print(f"[ok] wrote {n} pairs -> {out}")
    print("[next] eyeball 10 lines, then: firectl dataset create secfix-v1 data/train_secfix.jsonl")
    print("[warn] CVEfixes overlaps real CVEs. Hold CWEval out as eval; it is a SEPARATE set, keep it that way.")


if __name__ == "__main__":
    main()
