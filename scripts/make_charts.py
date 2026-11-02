"""README charts as dependency-free SVG (light + dark), generated from results/.

docs/img/method[-dark].svg              the pipeline (data factory -> Fireworks SFT -> review harness -> eval)
docs/img/before_after[-dark].svg        base -> + fine-tune -> + review, with frontier reference lines
docs/img/secure_rate[-dark].svg         works+secure by system
docs/img/quality_vs_latency[-dark].svg  works+secure vs median latency (log scale)
docs/img/per_language[-dark].svg        base vs CyberNemo vs CyberNemo + review, per language
Usage: python scripts/make_charts.py
"""
import json, math, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "docs", "img")
THEMES = {  # categorical slots validated with the dataviz palette checker (CVD + contrast; values are labelled)
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e7e6e2", box="#f1f0ec",
                  base="#eb6834", ours="#2a78d6", review="#1baf7a", other="#b5b3ad"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#33332f", box="#262624",
                 base="#d95926", ours="#3987e5", review="#199e70", other="#6b6a65"),
}
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"
BASE, SFT, REVIEW = "base_wandb_all_n5", "sft120_think_v2_n10_thinkon", "sft120_think_v2_n10_thinkon_review"
# (label, run, role, latency file, latency note)
SYSTEMS = [
    ("Reflection Beam 1.0", "beam_n10", "other", "beam_speed", ""),
    ("Gemini 3.8 Flash", "gemini38flash_n10", "other", "gemini38flash_speed", ""),
    ("CyberNemo SFT + harness (ours)", REVIEW, "review", "sft120_think_v2_n10_thinkon_speed", "x2"),  # two thinking calls
    ("DeepSeek V4.1 Flash", "dsv41flash_n10", "other", "dsv41flash_speed", ""),
    ("Gemini 3.5 Flash", "gemini35flash_all_n5", "other", "gemini35flash", ""),
    ("CyberNemo SFT (ours)", SFT, "ours", "sft120_think_v2_n10_thinkon_speed", ""),
    ("MiniMax M3.1 Flash", "minimax31flash_n5", "other", None, ""),
    ("Nemotron 120B base", BASE, "base", "nemotron120b_think", ""),
]


def pt(run):
    return {k: v for k, v in json.load(open(os.path.join(RES, run, "summary.json")))["per_task"].items() if "/go/" not in k}


def rate(run, lang=None, metric="func_sec"):  # CWEval pass@1: mean over tasks of the per-task pass rate
    rows = [v for k, v in pt(run).items() if lang is None or k.split("/")[1] == lang]
    return 100 * sum(v[metric] / v["n"] for v in rows) / len(rows)


def latency(name):
    return json.load(open(os.path.join(RES, "speed", f"{name}.json")))["summary"]["median_latency_s"]


def svg(w, h, body, t, title):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' {FONT}>"
            f"<title>{title}</title><rect width='{w}' height='{h}' rx='8' fill='{t['bg']}'/>{body}</svg>")


def hbar(x, y, w, h, fill):  # 4px rounded data end, square at the baseline
    r = min(4, w / 2)
    return (f"<path d='M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} "
            f"Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z' fill='{fill}'/>")


def vbar(x, y0, y, w, fill):
    r = min(4, w / 2)
    return (f"<path d='M{x},{y0} V{y + r} Q{x},{y} {x + r},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y0} Z' "
            f"fill='{fill}'/>")


def title(t, text, sub=None):
    s = f"<text x='20' y='28' font-size='15' font-weight='600' fill='{t['ink']}'>{text}</text>"
    if sub:
        s += f"<text x='20' y='46' font-size='12' fill='{t['ink2']}'>{sub}</text>"
    return s


def method(t):
    W, H = 820, 360
    b = [title(t, "How CyberNemo works: two parts", "Part 1 changes the model (training). Part 2 wraps it at answer time (harness).")]
    panels = [
        (20, "ours", "1  SFT fine-tune", "teaches secure habits  (+4.9 works+secure)",
         ["Practice tasks: 1,813 coding tasks with attack tests", "(written by Nemotron 3 Ultra, verified in a sandbox)",
          "Self-distillation: Nemotron 120B solves them;", "keep only its answers that pass every test",
          "LoRA SFT on Fireworks  &#8594;  CyberNemo"]),
        (420, "review", "2  Security-review harness", "catches what is left  (+8.9 works+secure)",
         ["CyberNemo writes the code", "",
          "Same model reviews it against a security checklist", "(escape HTML, reject CR/LF, check paths and URLs, ...)",
          "Returns the fixed code; never sees benchmark tests"]),
    ]
    for x, role, head, sub, lines in panels:
        b.append(f"<rect x='{x}' y='64' width='380' height='230' rx='12' fill='{t['box']}' stroke='{t[role]}' stroke-width='2'/>"
                 f"<text x='{x + 18}' y='94' font-size='15' font-weight='700' fill='{t['ink']}'>{head}</text>"
                 f"<text x='{x + 18}' y='114' font-size='12' fill='{t['ink2']}'>{sub}</text>")
        for i, pair in enumerate((lines[0:2], lines[2:4], lines[4:5])):
            y = 136 + i * 52
            b.append(f"<rect x='{x + 18}' y='{y}' width='344' height='{44 if len(pair) == 2 else 30}' rx='8' fill='{t['bg']}' stroke='{t['grid']}'/>")
            for j, line in enumerate(pair):
                if line:
                    b.append(f"<text x='{x + 30}' y='{y + 18 + j * 17}' font-size='12' fill='{t['ink'] if j == 0 else t['ink2']}'>{line}</text>")
            if i < 2:
                b.append(f"<path d='M{x + 190},{y + 44} V{y + 52}' stroke='{t['muted']}' stroke-width='2'/>")
    b.append(f"<path d='M403,180 H415' stroke='{t['muted']}' stroke-width='2'/><path d='M409,174 L417,180 L409,186' fill='none' stroke='{t['muted']}' stroke-width='2'/>")
    b.append(f"<rect x='20' y='308' width='780' height='36' rx='10' fill='{t['box']}' stroke='{t['grid']}'/>"
             f"<text x='36' y='331' font-size='12' fill='{t['ink2']}'><tspan font-weight='600' fill='{t['ink']}'>Evaluation:</tspan> "
             f"CWEval: 100 held-out human-written tasks, functionality + attack tests, run offline</text>")
    return svg(W, H, "".join(b), t, "CyberNemo: SFT fine-tune + security-review harness")


def before_after(t):
    stages = [("Nemotron 120B base", BASE, "base"), ("+ SFT fine-tune", SFT, "ours"), ("+ review harness", REVIEW, "review")]
    refs = [("Reflection Beam", "beam_n10"), ("Gemini 3.8 Flash", "gemini38flash_n10"), ("DeepSeek V4.1 Flash", "dsv41flash_n10"), ("Gemini 3.5 Flash", "gemini35flash_all_n5")]
    W, H, L, T, B = 820, 380, 56, 70, 74
    sy = lambda v: T + (H - T - B) * (1 - (v - 40) / 50)
    b = [title(t, "Before vs after: working AND secure code on CWEval (%)", "Thinking mode, same 100 held-out tasks; dashed lines = frontier models (default mode)")]
    for v in range(40, 91, 10):
        b.append(f"<line x1='{L}' y1='{sy(v)}' x2='{W - 190}' y2='{sy(v)}' stroke='{t['grid']}'/>"
                 f"<text x='{L - 8}' y='{sy(v) + 4}' font-size='11' fill='{t['muted']}' text-anchor='end'>{v}</text>")
    for name, run in refs:
        v = rate(run)
        b.append(f"<line x1='{L}' y1='{sy(v)}' x2='{W - 190}' y2='{sy(v)}' stroke='{t['muted']}' stroke-dasharray='5 4'/>"
                 f"<text x='{W - 184}' y='{sy(v) + 4}' font-size='11' fill='{t['ink2']}'>{name} {v:.1f}</text>")
    gw, bw = (W - 190 - L) / 3, 96
    deltas = ["baseline", "+4.9 (95% CI +0.6 to +9.4)", "+8.9 (95% CI +1.7 to +16.3)"]
    for i, (name, run, role) in enumerate(stages):
        v, works = rate(run), rate(run, metric="func")
        x = L + gw * i + (gw - bw) / 2
        b.append(vbar(x, sy(40), sy(v), bw, t[role]))  # value inside the bar, so reference lines never cross it
        b.append(f"<text x='{x + bw / 2}' y='{sy(v) + 22}' font-size='15' font-weight='700' fill='#ffffff' text-anchor='middle'>{v:.1f}</text>"
                 f"<text x='{x + bw / 2}' y='{H - B + 18}' font-size='12' font-weight='600' fill='{t['ink']}' text-anchor='middle'>{name}</text>"
                 f"<text x='{x + bw / 2}' y='{H - B + 35}' font-size='11' fill='{t['ink2']}' text-anchor='middle'>{deltas[i]}</text>"
                 f"<text x='{x + bw / 2}' y='{H - B + 51}' font-size='11' fill='{t['muted']}' text-anchor='middle'>works {works:.1f}</text>")
    return svg(W, H, "".join(b), t, "Before vs after")


def secure_rate(t):
    rows = sorted(((lbl, rate(run), role) for lbl, run, role, _, _ in SYSTEMS), key=lambda r: -r[1])
    W, L, top, bh, gap = 780, 240, 54, 22, 12
    H = top + len(rows) * (bh + gap) + 34
    sx = lambda v: L + (W - L - 70) * v / 100
    b = [title(t, "Working and secure code, CWEval (% of answers)")]
    for v in range(0, 101, 20):
        b.append(f"<line x1='{sx(v)}' y1='{top - 6}' x2='{sx(v)}' y2='{H - 28}' stroke='{t['grid']}'/>"
                 f"<text x='{sx(v)}' y='{H - 12}' font-size='11' fill='{t['muted']}' text-anchor='middle'>{v}</text>")
    for i, (lbl, v, role) in enumerate(rows):
        y = top + i * (bh + gap)
        mine = role in ("ours", "review")
        b.append(f"<text x='{L - 10}' y='{y + bh / 2 + 4}' font-size='13' fill='{t['ink'] if role != 'other' else t['ink2']}' "
                 f"text-anchor='end' font-weight='{600 if mine else 400}'>{lbl}</text>")
        b.append(hbar(L, y, sx(v) - L, bh, t[role]))
        b.append(f"<text x='{sx(v) + 6}' y='{y + bh / 2 + 4}' font-size='12' fill='{t['ink']}'>{v:.1f}</text>")
    return svg(W, H, "".join(b), t, "Working and secure code rate by system")


def quality_vs_latency(t):
    pts = []
    for lbl, run, role, sp, note in SYSTEMS:
        if sp is None:
            continue
        s = latency(sp) * (2 if note == "x2" else 1)
        pts.append((lbl + (" (2 calls, est.)" if note == "x2" else ""), s, rate(run), role, False))
    pts += [("CyberNemo, fast mode", latency("sft120_think_v2_n10_speed_run2"), rate("sft120_think_v1_n10"), "ours", True),
            ("Nemotron base, fast mode", latency("nemotron120b_nothink"), rate("super120_nothink_n10"), "base", True)]
    W, H, L, R, T, B = 820, 440, 64, 30, 56, 56
    lo, hi = math.log10(0.5), math.log10(30)
    sx = lambda s: L + (W - L - R) * (math.log10(s) - lo) / (hi - lo)
    sy = lambda v: T + (H - T - B) * (1 - (v - 45) / 35)
    b = [title(t, "Security vs speed", "Up = more secure; left = faster. Hollow = fast mode (no reasoning)")]
    for v in range(45, 81, 5):
        b.append(f"<line x1='{L}' y1='{sy(v)}' x2='{W - R}' y2='{sy(v)}' stroke='{t['grid']}'/>"
                 f"<text x='{L - 8}' y='{sy(v) + 4}' font-size='11' fill='{t['muted']}' text-anchor='end'>{v}</text>")
    for s in (0.5, 1, 2, 5, 10, 20, 30):
        b.append(f"<text x='{sx(s)}' y='{H - B + 18}' font-size='11' fill='{t['muted']}' text-anchor='middle'>{s:g} s</text>")
    b.append(f"<text x='{(L + W - R) / 2}' y='{H - 12}' font-size='12' fill='{t['ink2']}' text-anchor='middle'>median latency per answer (log scale)</text>"
             f"<text x='16' y='{(T + H - B) / 2}' font-size='12' fill='{t['ink2']}' text-anchor='middle' transform='rotate(-90 16 {(T + H - B) / 2})'>works + secure (%)</text>")
    nudge = {"CyberNemo SFT + harness (ours) (2 calls, est.)": 3, "DeepSeek V4.1 Flash": 17,
             "CyberNemo, fast mode": -12, "Nemotron 120B base": 12}
    for lbl, s, v, role, hollow in pts:
        x, y = sx(s), sy(v)
        b.append(f"<circle cx='{x}' cy='{y}' r='6' fill='{t['bg'] if hollow else t[role]}' stroke='{t[role] if hollow else t['bg']}' stroke-width='2'/>")
        anchor, dx = ("start", 10) if x < W - 260 else ("end", -10)
        b.append(f"<text x='{x + dx}' y='{y + 4 + nudge.get(lbl, 0)}' font-size='12' fill='{t['ink'] if role != 'other' else t['ink2']}' text-anchor='{anchor}'>{lbl}</text>")
    return svg(W, H, "".join(b), t, "Security versus latency")


def per_language(t):
    langs = [("C", "c"), ("C++", "cpp"), ("JavaScript", "js"), ("Python", "py")]
    series = [("Nemotron 120B base", BASE, "base"), ("+ SFT fine-tune", SFT, "ours"), ("+ review harness", REVIEW, "review")]
    W, H, L, T, B = 760, 340, 50, 76, 40
    gw, bw = (W - L - 30) / len(langs), 34
    sy = lambda v: T + (H - T - B) * (1 - v / 100)
    b = [title(t, "Working and secure code by language (thinking mode)")]
    for i, (lbl, _, role) in enumerate(series):  # legend
        x = 24 + i * 180
        b.append(f"<rect x='{x}' y='44' width='12' height='12' rx='3' fill='{t[role]}'/>"
                 f"<text x='{x + 18}' y='54' font-size='12' fill='{t['ink2']}'>{lbl}</text>")
    for v in range(0, 101, 25):
        b.append(f"<line x1='{L}' y1='{sy(v)}' x2='{W - 30}' y2='{sy(v)}' stroke='{t['grid']}'/>"
                 f"<text x='{L - 8}' y='{sy(v) + 4}' font-size='11' fill='{t['muted']}' text-anchor='end'>{v}</text>")
    for gi, (name, code) in enumerate(langs):
        cx = L + gw * gi + gw / 2
        for si, (_, run, role) in enumerate(series):
            v = rate(run, code)
            x = cx - 1.5 * bw - 2 + si * (bw + 2)
            b.append(vbar(x, sy(0), sy(v), bw, t[role]) +
                     f"<text x='{x + bw / 2}' y='{sy(v) - 6}' font-size='11' fill='{t['ink']}' text-anchor='middle'>{v:.0f}</text>")
        b.append(f"<text x='{cx}' y='{H - 16}' font-size='12' fill='{t['ink2']}' text-anchor='middle'>{name}</text>")
    return svg(W, H, "".join(b), t, "Working and secure code by language")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("method", method), ("before_after", before_after), ("secure_rate", secure_rate),
                     ("quality_vs_latency", quality_vs_latency), ("per_language", per_language)):
        for mode, t in THEMES.items():
            path = os.path.join(OUT, f"{name}{'' if mode == 'light' else '-dark'}.svg")
            open(path, "w").write(fn(t))
            print("wrote", os.path.relpath(path, ROOT))
