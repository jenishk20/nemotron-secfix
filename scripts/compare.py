"""Results tables and paired confidence intervals from saved CWEval summaries.

Reads results/<run>/summary.json (copied from outputs/<run>/ by --collect) so judges can
re-derive every number in the README without re-running models. Go tasks are excluded
everywhere (our harness has no Go toolchain parity), leaving 100 tasks.
Paired test: per-task difference in pass rate between two systems, bootstrap over the
100 tasks (10,000 resamples), 95% CI.
Usage: python scripts/compare.py [--collect] [--vs <base_run> <other_run> ...]
"""
import argparse, collections, json, os, random, shutil, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
RUNS = [  # (label, run, how it was produced)
    ("Gemini 3.8 Flash", "gemini38flash_n10", "Vertex AI, default mode"),
    ("Reflection Beam 1.0 501B", "beam_n10", "W&B Inference, default mode"),
    ("DeepSeek V4.1 Flash", "dsv41flash_n10", "W&B Inference, default mode"),
    ("Gemini 3.5 Flash", "gemini35flash_all_n5", "Vertex AI, default mode"),
    ("Nemotron 3.5 Super 120B, base (thinking off)", "super120_nothink_n10", "W&B Inference"),
    ("Nemotron 3.5 Super 120B, base + security prompt", "super120_nothink_secure_n5", "W&B Inference"),
    ("Nemotron 3.5 Super 120B, base (thinking on)", "base_wandb_all_n5", "W&B Inference"),
    ("**Ours: Nemotron 120B CyberNemo v1 (thinking off)**", "sft120_think_v1_n10", "Fireworks LoRA, dedicated deployment"),
    ("**Ours: Nemotron 120B CyberNemo v2 (thinking off)**", "sft120_think_v2_n10", "Fireworks LoRA, dedicated deployment"),
    ("**Ours: Nemotron 120B CyberNemo v2 (thinking on)**", "sft120_think_v2_n10_thinkon", "Fireworks LoRA, dedicated deployment"),
    ("**Ours: CyberNemo v3 (thinking)**", "sftv3_thinkon", "Fireworks LoRA, dedicated deployment"),
    ("**Ours: CyberNemo v2 (thinking) + security review**", "sft120_think_v2_n10_thinkon_review", "Fireworks LoRA + scripts/review_pass.py (869 answers)"),
    ("Nemotron 3.5 Lightning 30B, base", "lightning_nothink_n10", "W&B Inference"),
]


def per_task(run):
    pt = json.load(open(os.path.join(RES, run, "summary.json")))["per_task"]
    return {k: v for k, v in pt.items() if "/go/" not in k}


def table():
    print("| System | Answers/task | Works | Works+secure | C | C++ | JS | Python |\n|---|---|---|---|---|---|---|---|")
    for label, run, _ in RUNS:
        if not os.path.exists(os.path.join(RES, run, "summary.json")):
            continue
        agg = collections.defaultdict(lambda: [0, 0.0, 0.0])  # CWEval pass@1: mean over tasks of per-task rate
        pt = per_task(run)
        for k, v in pt.items():
            lang = k.split("/")[1]
            for key in (lang, "all"):
                a = agg[key]; a[0] += 1; a[1] += v["func"] / v["n"]; a[2] += v["func_sec"] / v["n"]
        f = lambda k, i: f"{100 * agg[k][i] / agg[k][0]:.1f}" if agg[k][0] else "-"
        n = round(sum(v["n"] for v in pt.values()) / len(pt))
        print(f"| {label} | {n} | {f('all', 1)} | {f('all', 2)} | {f('c', 2)} | {f('cpp', 2)} | {f('js', 2)} | {f('py', 2)} |")
    print("\nPer-language columns are works+secure; C pools CWEval's core/c and lang/c suites.")


def paired(base, other, metric="func_sec", reps=10000):
    b, o = per_task(base), per_task(other)
    d = [o[k][metric] / o[k]["n"] - b[k][metric] / b[k]["n"] for k in b]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(reps))
    return 100 * sum(d) / len(d), 100 * boots[int(0.025 * reps)], 100 * boots[int(0.975 * reps)], \
        sum(x > 0 for x in d), sum(x < 0 for x in d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--vs", nargs="+")
    a = ap.parse_args()
    if a.collect:
        for _, run, _ in RUNS:
            src = os.path.join(ROOT, "outputs", run, "summary.json")
            if os.path.exists(src) and run != "sft120_think_v2_n10_thinkon_review":  # rebuilt by summarize_existing.py
                os.makedirs(os.path.join(RES, run), exist_ok=True)
                shutil.copy(src, os.path.join(RES, run, "summary.json"))
    table()
    if a.vs:
        base = a.vs[0]
        for other in a.vs[1:]:
            for m in ("func_sec", "func"):
                mean, lo, hi, up, down = paired(base, other, m)
                print(f"{other} vs {base} [{'works+secure' if m == 'func_sec' else 'works'}]: "
                      f"{mean:+.1f} pts (95% CI {lo:+.1f} to {hi:+.1f}); tasks better {up}, worse {down}")
