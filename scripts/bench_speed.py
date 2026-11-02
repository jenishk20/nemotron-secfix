"""Speed / token / cost benchmark on CWEval prompts (the `direct` prompt, one answer per task).

Streams each answer and records time to first token, total latency, output tokens and output
tokens/s, plus the price per task from the given $/1M token prices. Runs a fixed, seeded subset of
the 100 non-Go CWEval tasks with modest concurrency so latency reflects one user, not a batch.
Model forms (same as scripts/gen.sh): wandb:<id>, vertex_ai/<id>, fireworks:<model or deployment>.
Usage: .venv/bin/python scripts/bench_speed.py <label> <model> [--thinking] [--n 30]
         [--price_in 0.15 --price_out 0.50]   -> outputs/speed/<label>.json
"""
import argparse, base64, glob, json, os, random, statistics, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "third_party", "CWEval"))
from cweval.ppt import DirectPrompt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
import litellm  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
BENCH = os.path.join(ROOT, "third_party", "CWEval", "benchmark")


def tasks(n, seed=0):
    out = []
    for p in sorted(glob.glob(os.path.join(BENCH, "*", "*", "*_task.*"))):
        lang = os.path.splitext(p)[1][1:]
        if lang == "go":
            continue
        src = open(p).read()
        sol = next(l for l in src.splitlines() if "BEGIN SOLUTION" in l)
        stub = src.split("BEGIN PROMPT")[-1].split(sol)[0].strip()
        out.append((os.path.relpath(p, BENCH), DirectPrompt.PPT.format(
            lang=lang, lang_instr=DirectPrompt.LANG_INSTR[lang], code_prompt=stub)))
    return random.Random(seed).sample(out, min(n, len(out)))


def route(model, thinking):
    kw = {}
    if model.startswith("wandb:"):
        kw = dict(model="openai/" + model[6:], api_base="https://api.inference.wandb.ai/v1",
                  api_key=os.environ["WANDB_API_KEY"],
                  extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
    elif model.startswith("fireworks:"):
        kw = dict(model="openai/" + model[10:], api_base="https://api.fireworks.ai/inference/v1",
                  api_key=os.environ["FIREWORKS_API_KEY"],
                  extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
    elif model.startswith("vertex_ai/"):
        adc = tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False)
        adc.write(base64.b64decode(os.environ["GOOGLE_ADC_B64"])); adc.close()
        os.chmod(adc.name, 0o600)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = adc.name
        kw = dict(model=model, vertex_project=os.environ["GCP_PROJECT"], vertex_location=os.environ["GCP_LOCATION"])
    else:
        kw = dict(model=model)
    return kw


def one(kw, prompt, max_tokens):
    t0, ttft, text, usage = time.time(), None, "", None
    for attempt in range(4):
        try:
            stream = litellm.completion(messages=[{"role": "user", "content": prompt}], stream=True,
                                        stream_options={"include_usage": True}, temperature=0.8,
                                        max_tokens=max_tokens, **kw)
            for ch in stream:
                if ch.choices and (ch.choices[0].delta.content or getattr(ch.choices[0].delta, "reasoning_content", None)):
                    ttft = ttft or time.time() - t0
                    text += ch.choices[0].delta.content or ""
                if getattr(ch, "usage", None):
                    usage = ch.usage
            break
        except Exception as e:
            if attempt == 3:
                return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            time.sleep(5 * (attempt + 1)); t0 = time.time()
    total = time.time() - t0
    out_tok = getattr(usage, "completion_tokens", None) or 0
    return {"ttft_s": ttft, "latency_s": total, "in_tokens": getattr(usage, "prompt_tokens", None) or 0,
            "out_tokens": out_tok, "answer_chars": len(text),
            "out_tok_per_s": out_tok / max(1e-6, total)}  # end to end: some providers buffer the stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("model")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--price_in", type=float, default=0)   # $ per 1M input tokens
    ap.add_argument("--price_out", type=float, default=0)  # $ per 1M output tokens
    a = ap.parse_args()
    kw = route(a.model, a.thinking)
    ts = tasks(a.n)
    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(lambda t: dict(task=t[0], **one(kw, t[1], a.max_tokens)), ts))
    ok = [r for r in res if "error" not in r]
    med = lambda k: statistics.median(r[k] for r in ok if r.get(k) is not None)
    summ = {"label": a.label, "model": a.model, "thinking": a.thinking, "tasks": len(ts), "errors": len(res) - len(ok),
            "median_ttft_s": round(med("ttft_s"), 2), "median_latency_s": round(med("latency_s"), 2),
            "p90_latency_s": round(sorted(r["latency_s"] for r in ok)[int(0.9 * (len(ok) - 1))], 2),
            "mean_out_tokens": round(statistics.mean(r["out_tokens"] for r in ok)),
            "median_out_tok_per_s": round(med("out_tok_per_s"), 1)}
    if a.price_in or a.price_out:
        summ["usd_per_1k_tasks"] = round(1000 * statistics.mean(
            r["in_tokens"] * a.price_in / 1e6 + r["out_tokens"] * a.price_out / 1e6 for r in ok), 3)
    os.makedirs(os.path.join(ROOT, "outputs", "speed"), exist_ok=True)
    json.dump({"summary": summ, "requests": res}, open(os.path.join(ROOT, "outputs", "speed", f"{a.label}.json"), "w"), indent=1)
    print(json.dumps(summ))
    if len(ok) < len(res):
        print("first error:", next(r["error"] for r in res if "error" in r))


if __name__ == "__main__":
    main()
