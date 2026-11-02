"""Fast CWEval answer generation for a short-lived (paid) endpoint, e.g. a Fireworks deployment.

Same prompts and sampling as scripts/gen.sh (CWEval `direct` prompt, temperature 0.8, 8192 max
tokens, thinking off), but every request runs in parallel (--concurrency), one answer per request,
so 100 tasks x 10 answers finish in minutes. Writes CWEval's layout
(outputs/<run>/generated_<i>/<suite>/<lang>/<task>_raw.<ext>), so scripts/score_sandbox.py scores it
unchanged. Resumable: existing non-empty answers are skipped.
Model forms: wandb:<id> | wbtrain:<wandb-artifact:///...> | fireworks:<id> | vertex_ai/<gemini id> (use .venv)
Usage: .venv-art/bin/python scripts/fast_gen.py <run> <model> [--n 10] [--concurrency 64] [--langs py,c,cpp,js]
         [--default_mode --max_tokens 32768]   (baselines: keep the model's default reasoning mode)
"""
import argparse, asyncio, glob, os, random, sys, time

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CWEval"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from cweval.ppt import DirectPrompt  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
BENCH = os.path.join(ROOT, "third_party", "CWEval", "benchmark")


def tasks(langs):
    out = []
    for p in sorted(glob.glob(os.path.join(BENCH, "*", "*", "*_task.*"))):
        lang = os.path.splitext(p)[1][1:]
        if lang not in langs:
            continue
        src = open(p).read()
        sol = next(l for l in src.splitlines() if "BEGIN SOLUTION" in l)
        stub = src.split("BEGIN PROMPT")[-1].split(sol)[0].strip()
        rel = os.path.relpath(p, BENCH).replace("_task.", "_raw.")  # core/py/cwe_020_0_raw.py
        out.append((rel, DirectPrompt.PPT.format(lang=lang, lang_instr=DirectPrompt.LANG_INSTR[lang], code_prompt=stub)))
    return out


async def main(a):
    if a.model.startswith("wandb:"):
        client, model = AsyncOpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=os.environ["WANDB_API_KEY"]), a.model[6:]
    elif a.model.startswith("wbtrain:"):  # a W&B Serverless Training checkpoint (wandb-artifact:///...)
        client, model = AsyncOpenAI(base_url="https://api.training.wandb.ai/v1/", api_key=os.environ["WANDB_API_KEY"]), a.model[8:]
    elif a.model.startswith("vertex_ai/"):  # Gemini on Vertex via litellm (run with .venv, which has litellm)
        import litellm
        from bench_speed import route
        kw = route(a.model, False)

        class _Vertex:  # minimal stand-in exposing client.chat.completions.create(...)
            class chat:
                class completions:
                    @staticmethod
                    async def create(model, messages, max_tokens, temperature, **_):
                        return await litellm.acompletion(messages=messages, max_tokens=max_tokens,
                                                         temperature=temperature, **kw)
        client, model = _Vertex(), a.model
    elif a.model.startswith("fireworks:"):
        client, model = AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"]), a.model[10:]
    else:
        raise SystemExit("model must start with wandb:, wbtrain:, vertex_ai/ or fireworks:")
    run_dir = os.path.join(ROOT, "outputs", a.run)
    jobs = [(os.path.join(run_dir, f"generated_{i}", rel), prompt) for rel, prompt in tasks(a.langs.split(","))
            for i in range(a.n)]
    jobs = [j for j in jobs if not (os.path.exists(j[0]) and os.path.getsize(j[0]))]
    print(f"{a.run}: {len(jobs)} answers to generate with {model}", flush=True)
    sem, done, failed, t0 = asyncio.Semaphore(a.concurrency), 0, 0, time.time()

    async def one(path, prompt):
        nonlocal done, failed
        for attempt in range(6):
            try:
                async with sem:
                    r = await client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.8,
                        max_tokens=a.max_tokens, **({} if a.default_mode else
                                                    {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}))
                text = r.choices[0].message.content or ""
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w").write(text)
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
                return
            except Exception as e:
                if attempt == 5:
                    failed += 1
                    print(f"  FAILED {os.path.relpath(path, run_dir)}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    return
                await asyncio.sleep(min(30, 2 ** attempt) + random.random())

    await asyncio.gather(*[one(p, q) for p, q in jobs])
    print(f"{a.run}: {done} written, {failed} failed in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("model")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--langs", default="py,c,cpp,js")
    ap.add_argument("--default_mode", action="store_true")  # frontier baselines: the model's own default (no thinking switch)
    ap.add_argument("--max_tokens", type=int, default=8192)
    asyncio.run(main(ap.parse_args()))
