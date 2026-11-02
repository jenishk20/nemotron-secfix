"""README charts as dependency-free SVG (light + dark), generated from results/.

docs/img/secure_rate[-dark].svg         works+secure by system (thinking / default mode)
docs/img/quality_vs_latency[-dark].svg  works+secure vs median latency (log scale)
docs/img/per_language[-dark].svg        base vs CyberNemo per language (thinking mode)
Usage: python scripts/make_charts.py
"""
import json, math, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "docs", "img")
THEMES = {  # validated with the dataviz palette checker (blue/orange pass CVD + contrast in both modes)
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e7e6e2",
                  ours="#2a78d6", base="#eb6834", other="#b5b3ad"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#33332f",
                 ours="#3987e5", base="#d95926", other="#6b6a65"),
}
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"
# (label, run, role, latency run) - thinking/default-mode systems
SYSTEMS = [
    ("Reflection Beam 1.0", "beam_n10", "other", "beam_speed"),
    ("Gemini 3.8 Flash", "gemini38flash_n10", "other", "gemini38flash_speed"),
    ("DeepSeek V4.1 Flash", "dsv41flash_n10", "other", "dsv41flash_speed"),
    ("Gemini 3.5 Flash", "gemini35flash_all_n5", "other", "gemini35flash"),
    ("CyberNemo (ours)", "sft120_think_v2_n10_thinkon", "ours", None),  # latency: estimated from base thinking mode
    ("Nemotron 120B base", "base_wandb_all_n5", "base", "nemotron120b_think"),
]


def rate(run, lang=None):
    pt = json.load(open(os.path.join(RES, run, "summary.json")))["per_task"]
    rows = [v for k, v in pt.items() if "/go/" not in k and (lang is None or k.split("/")[1] == lang)]
    return 100 * sum(v["func_sec"] for v in rows) / sum(v["n"] for v in rows)


def latency(name):
    return json.load(open(os.path.join(RES, "speed", f"{name}.json")))["summary"]["median_latency_s"]


def svg(w, h, body, t, title):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' {FONT}>"
            f"<title>{title}</title><rect width='{w}' height='{h}' rx='8' fill='{t['bg']}'/>{body}</svg>")


def bar(x, y, w, h, fill):  # 4px rounded data end, square at the baseline
    r = min(4, w / 2)
    return (f"<path d='M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} "
            f"Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z' fill='{fill}'/>")


def secure_rate(t):
    rows = sorted(((lbl, rate(run), role) for lbl, run, role, _ in SYSTEMS), key=lambda r: -r[1])
    W, L, top, bh, gap = 760, 190, 54, 22, 12
    H = top + len(rows) * (bh + gap) + 34
    sx = lambda v: L + (W - L - 70) * v / 100
    b = [f"<text x='20' y='28' font-size='15' font-weight='600' fill='{t['ink']}'>Working and secure code, CWEval (% of answers)</text>"]
    for v in range(0, 101, 20):
        b.append(f"<line x1='{sx(v)}' y1='{top - 6}' x2='{sx(v)}' y2='{H - 28}' stroke='{t['grid']}'/>"
                 f"<text x='{sx(v)}' y='{H - 12}' font-size='11' fill='{t['muted']}' text-anchor='middle'>{v}</text>")
    for i, (lbl, v, role) in enumerate(rows):
        y = top + i * (bh + gap)
        b.append(f"<text x='{L - 10}' y='{y + bh / 2 + 4}' font-size='13' fill='{t['ink'] if role != 'other' else t['ink2']}' "
                 f"text-anchor='end' font-weight='{600 if role == 'ours' else 400}'>{lbl}</text>")
        b.append(bar(L, y, sx(v) - L, bh, t[role]))
        b.append(f"<text x='{sx(v) + 6}' y='{y + bh / 2 + 4}' font-size='12' fill='{t['ink']}'>{v:.1f}</text>")
    return svg(W, H, "".join(b), t, "Working and secure code rate by system")


def quality_vs_latency(t):
    pts = [(lbl if sp else lbl + " (latency est.)", latency(sp) if sp else latency("nemotron120b_think"), rate(run), role)
           for lbl, run, role, sp in SYSTEMS]
    pts += [("CyberNemo, fast mode", latency("sft120_think_v2_n10_speed"), rate("sft120_think_v1_n10"), "ours"),
            ("Nemotron base, fast mode", latency("nemotron120b_nothink"), rate("super120_nothink_n10"), "base")]
    W, H, L, R, T, B = 760, 430, 64, 30, 50, 56
    lo, hi = math.log10(0.5), math.log10(30)
    sx = lambda s: L + (W - L - R) * (math.log10(s) - lo) / (hi - lo)
    sy = lambda v: T + (H - T - B) * (1 - (v - 45) / 35)
    b = [f"<text x='20' y='28' font-size='15' font-weight='600' fill='{t['ink']}'>Security vs speed: up and to the left is better</text>"]
    for v in range(45, 81, 5):
        b.append(f"<line x1='{L}' y1='{sy(v)}' x2='{W - R}' y2='{sy(v)}' stroke='{t['grid']}'/>"
                 f"<text x='{L - 8}' y='{sy(v) + 4}' font-size='11' fill='{t['muted']}' text-anchor='end'>{v}</text>")
    for s in (0.5, 1, 2, 5, 10, 20, 30):
        b.append(f"<text x='{sx(s)}' y='{H - B + 18}' font-size='11' fill='{t['muted']}' text-anchor='middle'>{s:g} s</text>")
    b.append(f"<text x='{(L + W - R) / 2}' y='{H - 12}' font-size='12' fill='{t['ink2']}' text-anchor='middle'>median latency per answer (log scale)</text>"
             f"<text x='16' y='{(T + H - B) / 2}' font-size='12' fill='{t['ink2']}' text-anchor='middle' transform='rotate(-90 16 {(T + H - B) / 2})'>works + secure (%)</text>")
    for lbl, s, v, role in pts:
        x, y = sx(s), sy(v)
        hollow = "fast mode" in lbl
        b.append(f"<circle cx='{x}' cy='{y}' r='6' fill='{t['bg'] if hollow else t[role]}' stroke='{t[role] if hollow else t['bg']}' stroke-width='2'/>")
        anchor, dx = ("start", 10) if x < W - 200 else ("end", -10)
        b.append(f"<text x='{x + dx}' y='{y + 4}' font-size='12' fill='{t['ink'] if role != 'other' else t['ink2']}' text-anchor='{anchor}'>{lbl}</text>")
    return svg(W, H, "".join(b), t, "Security versus latency")


def per_language(t):
    langs = [("C", "c"), ("C++", "cpp"), ("JavaScript", "js"), ("Python", "py")]
    series = [("Nemotron 120B base", "base_wandb_all_n5", "base"), ("CyberNemo (ours)", "sft120_think_v2_n10_thinkon", "ours")]
    W, H, L, T, B = 760, 330, 50, 70, 40
    gw, bw = (W - L - 30) / len(langs), 34
    sy = lambda v: T + (H - T - B) * (1 - v / 100)
    b = [f"<text x='20' y='28' font-size='15' font-weight='600' fill='{t['ink']}'>Working and secure code by language (thinking mode)</text>"]
    for i, (lbl, _, role) in enumerate(series):  # legend
        x = 24 + i * 190
        b.append(f"<rect x='{x}' y='42' width='12' height='12' rx='3' fill='{t[role]}'/>"
                 f"<text x='{x + 18}' y='52' font-size='12' fill='{t['ink2']}'>{lbl}</text>")
    for v in range(0, 101, 25):
        b.append(f"<line x1='{L}' y1='{sy(v)}' x2='{W - 30}' y2='{sy(v)}' stroke='{t['grid']}'/>"
                 f"<text x='{L - 8}' y='{sy(v) + 4}' font-size='11' fill='{t['muted']}' text-anchor='end'>{v}</text>")
    for gi, (name, code) in enumerate(langs):
        cx = L + gw * gi + gw / 2
        for si, (_, run, role) in enumerate(series):
            v = rate(run, code)
            x = cx - bw - 1 + si * (bw + 2)
            y = sy(v)
            r = 4
            b.append(f"<path d='M{x},{sy(0)} V{y + r} Q{x},{y} {x + r},{y} H{x + bw - r} Q{x + bw},{y} {x + bw},{y + r} V{sy(0)} Z' fill='{t[role]}'/>"
                     f"<text x='{x + bw / 2}' y='{y - 6}' font-size='11' fill='{t['ink']}' text-anchor='middle'>{v:.0f}</text>")
        b.append(f"<text x='{cx}' y='{H - 16}' font-size='12' fill='{t['ink2']}' text-anchor='middle'>{name}</text>")
    return svg(W, H, "".join(b), t, "Working and secure code by language")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("secure_rate", secure_rate), ("quality_vs_latency", quality_vs_latency), ("per_language", per_language)):
        for mode, t in THEMES.items():
            path = os.path.join(OUT, f"{name}{'' if mode == 'light' else '-dark'}.svg")
            open(path, "w").write(fn(t))
            print("wrote", os.path.relpath(path, ROOT))
