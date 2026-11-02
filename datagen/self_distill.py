"""Self-distillation (rejection-sampling fine-tuning) data from the base model's own answers.

For every verified synthetic task (all languages, never CWEval), sample --k answers from the base
Nemotron 3.5 Lightning 30B-A3B on W&B Inference with CWEval's `secure` prompt (thinking off), run
each answer against the task's own tests in offline W&B Sandboxes (datagen/reward_runner.py), and keep
answers that pass every functionality and security test. The SFT pair is CWEval's plain `direct`
prompt -> the model's own passing answer, so the model learns to write secure code unprompted
while staying in its own style (no teacher-style drift).

Stages (each resumable, files under data/synth/sd_*; --tag keeps runs for different models apart):
  gen    sample answers             -> data/synth/sd[_<tag>]_samples.jsonl
  score  test them in W&B Sandboxes -> data/synth/sd[_<tag>]_scores.jsonl
  build  passing answers -> SFT     -> data/selfdistill[_<tag>]_train.jsonl
Usage: .venv-art/bin/python datagen/self_distill.py gen|score|build [--k 4] [--limit N]
         [--model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B --tag 120b] [--thinking --max_tokens 16384]
"""
import argparse, asyncio, glob, io, json, os, random, re, sys, tarfile, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "datagen"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CWEval"))
from build_sft import cweval_prompts, leaks  # noqa: E402
from cweval.ppt import DirectPrompt, SecurePrompt  # noqa: E402
from verify_in_container import PYTEST_INI  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"
ENTITY, PROJECT = os.environ["WANDB_ENTITY"], os.environ.get("WANDB_PROJECT", "nemotron-secfix")
ENV = ". /home/ubuntu/miniforge3/etc/profile.d/conda.sh && cd /home/ubuntu/CWEval && source .env"
S = os.path.join(ROOT, "data", "synth")


def load_tasks():
    """Every verified synthetic task (any language, any repair round), minus anything overlapping CWEval."""
    recs, bench = {}, cweval_prompts()
    for f in sorted(glob.glob(os.path.join(S, "*.jsonl"))):
        vf = f.replace(".jsonl", ".verify.jsonl")
        if f.endswith(".verify.jsonl") or not os.path.exists(vf) or "regress" in f or "/sd_" in f:
            continue
        ok = {r["id"] for r in map(json.loads, open(vf)) if r.get("ok")}
        for r in map(json.loads, open(f)):
            if r["id"] in ok and r["id"] not in recs and not leaks(r["stub"], bench):
                r.setdefault("lang", "py")
                recs[r["id"]] = r
    return sorted(recs.values(), key=lambda r: r["id"])


def prompt(task, ppt):
    return ppt.PPT.format(lang=task["lang"], lang_instr=ppt.LANG_INSTR[task["lang"]],
                          code_prompt=task["stub"].rstrip("\n"))


def first_code_block(text):
    m = re.search(r"```[a-zA-Z+]*\n(.*?)```", text or "", re.S)
    return m.group(1) if m else (text or "")


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


# ---------------------------------------------------------------- gen
async def gen(a, tasks):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=os.environ["WANDB_API_KEY"],
                         timeout=300)
    done = {(r["id"], r["i"]) for r in read_jsonl(SAMPLES)}
    todo = [(t, i) for t in tasks for i in range(a.k) if (t["id"], i) not in done]
    print(f"{len(tasks)} tasks x {a.k} samples: {len(todo)} to generate", flush=True)
    sem, out, n, t0 = asyncio.Semaphore(a.concurrency), open(SAMPLES, "a"), 0, time.time()

    async def one(t, i):
        nonlocal n
        for attempt in range(8):
            try:
                async with sem:
                    r = await client.chat.completions.create(
                        model=a.model, messages=[{"role": "user", "content": prompt(t, SecurePrompt)}],
                        temperature=0.8, max_tokens=a.max_tokens,
                        **({} if a.default_mode else {"extra_body": {"chat_template_kwargs": {"enable_thinking": a.thinking}}}))
                break
            except Exception as e:
                status = getattr(e, "status_code", None)
                if attempt == 7 or not (status in (408, 429) or (status and status >= 500)
                                        or "timeout" in type(e).__name__.lower()
                                        or "connection" in type(e).__name__.lower()):
                    print(f"  {t['id']}#{i} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
                    return
                await asyncio.sleep(min(60, 2 ** attempt) + random.random())
        # with thinking on, keep only the final answer (the SFT target is the answer without its reasoning)
        content = re.sub(r"^.*?</think>\s*", "", r.choices[0].message.content or "", flags=re.S)
        out.write(json.dumps({"id": t["id"], "lang": t["lang"], "i": i, "content": content,
                              "finish": r.choices[0].finish_reason,
                              "out_tokens": getattr(r.usage, "completion_tokens", None)}) + "\n")
        out.flush()
        n += 1
        if n % 250 == 0:
            print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    await asyncio.gather(*[one(t, i) for t, i in todo])
    out.close()
    print(f"generated {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- score
def score_chunk(ci, items, a):
    """items: [(sample_key, task, answer)] -> {sample_key: counts}, in one offline W&B Sandbox."""
    import cwsandbox as cw
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def add(path, text):
            data = text.encode()
            info = tarfile.TarInfo(path); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for j, (key, t, ans) in enumerate(items):
            d = f"chunk/s{j}"
            add(f"{d}/test_task.py", t["tests"]); add(f"{d}/pytest.ini", PYTEST_INI)
            if t["lang"] == "py":
                add(f"{d}/impl.py", first_code_block(ans))
            else:
                add(f"{d}/code.txt", first_code_block(ans))
                add(f"{d}/meta.json", json.dumps({"lang": t["lang"], "entrypoint": t["entrypoint"]}))
    t0 = time.time()
    sb = cw.Sandbox.run(container_image="co1lin/cweval", auth=cw.WandbAuth(entity=ENTITY, project=PROJECT),
                        network=cw.NetworkOptions(deny_egress=True, deny_ingress=True),  # model code stays offline
                        resources={"requests": {"cpu": a.cpu, "memory": a.memory},
                                   "limits": {"cpu": a.cpu, "memory": a.memory}},
                        max_lifetime_seconds=3600, tags=["secfix-selfdistill", f"chunk{ci}"])
    try:
        sb.wait(timeout=900)
        sb.exec(["mkdir", "-p", "/tmp/datagen", "/tmp/work"]).result()
        for f in ("verify_in_container.py", "reward_runner.py"):
            sb.write_file(f"/tmp/datagen/{f}", open(os.path.join(ROOT, "datagen", f), "rb").read()).result()
        sb.write_file("/tmp/chunk.tgz", buf.getvalue()).result()
        cmd = (f"{ENV} && tar -xzf /tmp/chunk.tgz -C /tmp/work && "
               f"python /tmp/datagen/reward_runner.py /tmp/work/chunk --workers {a.workers}")
        r = sb.exec(["bash", "-lc", cmd], timeout_seconds=3300).result()
        if r.returncode != 0:
            raise RuntimeError(f"rc={r.returncode}: {((r.stdout or '') + (r.stderr or ''))[-1500:]}")
        res = json.loads(sb.read_file("/tmp/work/chunk/results.json").result())
    finally:
        sb.stop().result()
    print(f"  chunk {ci}: {len(items)} scored in {time.time() - t0:.0f}s", flush=True)
    return {items[int(k[1:])][0]: v for k, v in res.items()}


def score(a, tasks):
    by_id = {t["id"]: t for t in tasks}
    done = {r["key"] for r in read_jsonl(SCORES)}
    todo = [(f"{s['id']}#{s['i']}", by_id[s["id"]], s["content"]) for s in read_jsonl(SAMPLES)
            if s["id"] in by_id and f"{s['id']}#{s['i']}" not in done]
    random.Random(0).shuffle(todo)  # mix languages so chunks take similar time
    chunks = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]
    print(f"{len(todo)} answers to score in {len(chunks)} sandboxes ({a.sandboxes} at a time)", flush=True)
    with open(SCORES, "a") as out, ThreadPoolExecutor(a.sandboxes) as ex:
        futs = {ex.submit(score_chunk, ci, c, a): ci for ci, c in enumerate(chunks)}
        for f in as_completed(futs):
            try:
                for key, c in f.result().items():
                    out.write(json.dumps({"key": key, **c}) + "\n")
                out.flush()
            except Exception as e:
                print(f"  chunk {futs[f]} FAILED (re-run score to retry): {type(e).__name__}: {str(e)[:300]}", flush=True)


# ---------------------------------------------------------------- build
def passes(c):
    return (not c["func_timeout"] and not c["sec_timeout"] and not c.get("compile_error")
            and c["func_fail"] == 0 and c["func_pass"] > 0 and c["sec_fail"] == 0 and c["sec_pass"] > 0)


def build(a, tasks):
    by_id = {t["id"]: t for t in tasks}
    scores = {r["key"]: r for r in read_jsonl(SCORES)}
    good, stats = defaultdict(list), defaultdict(Counter)
    for s in read_jsonl(SAMPLES):
        key, t = f"{s['id']}#{s['i']}", by_id.get(s["id"])
        if t is None or key not in scores:
            continue
        c = scores[key]
        st = stats[t["lang"]]
        st["scored"] += 1
        st["works"] += c["func_fail"] == 0 and c["func_pass"] > 0 and not c["func_timeout"] and not c.get("compile_error")
        st["compile_error"] += bool(c.get("compile_error"))
        text = s["content"].strip()
        one_block = text.startswith("```") and text.endswith("```") and text.count("```") == 2  # no prose around the code
        if passes(c) and s.get("finish") != "length" and one_block:
            st["works+secure"] += 1
            good[s["id"]].append(s["content"].strip())
    rows, kept = [], Counter()
    for tid, answers in sorted(good.items()):
        t, seen = by_id[tid], []
        for ans in answers:
            code = first_code_block(ans).strip()
            if code in seen:
                continue
            seen.append(code)
            rows.append({"messages": [{"role": "user", "content": prompt(t, DirectPrompt)},
                                      {"role": "assistant", "content": ans}]})
            kept[t["lang"]] += 1
            if len(seen) >= a.per_task:
                break
    random.Random(0).shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for lang, st in sorted(stats.items()):
        n = st["scored"]
        print(f"{lang}: {n} answers scored | works {st['works'] / n:.1%} | works+secure {st['works+secure'] / n:.1%} | "
              f"compile errors {st['compile_error'] / n:.1%} | tasks with a passing answer "
              f"{sum(1 for i in good if by_id[i]['lang'] == lang)} | SFT rows {kept[lang]}")
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "score", "build"])
    ap.add_argument("--k", type=int, default=4)              # answers per task
    ap.add_argument("--limit", type=int, default=0)          # only the first N tasks (smoke test)
    ap.add_argument("--concurrency", type=int, default=16)   # W&B Inference parallel requests
    ap.add_argument("--chunk", type=int, default=400)        # answers per sandbox
    ap.add_argument("--sandboxes", type=int, default=6)      # sandboxes at a time
    ap.add_argument("--cpu", default="8")
    ap.add_argument("--memory", default="16Gi")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per_task", type=int, default=2)       # SFT rows per task (distinct passing answers)
    ap.add_argument("--model", default=MODEL)               # W&B Inference model to distill from
    ap.add_argument("--tag", default="")                     # file suffix, e.g. 120b
    ap.add_argument("--thinking", action="store_true")       # sample with reasoning on; train on the final answer only
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--langs", default="")                   # e.g. c,cpp: only these languages
    ap.add_argument("--only", default="")                    # comma-separated task-id substrings, e.g. _901_,_902_
    ap.add_argument("--ids_file", default="")                # file with one task id per line: only these tasks
    ap.add_argument("--default_mode", action="store_true")   # teacher models that always reason (no thinking switch)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    SAMPLES, SCORES = os.path.join(S, f"sd{sfx}_samples.jsonl"), os.path.join(S, f"sd{sfx}_scores.jsonl")
    a.out = a.out or os.path.join(ROOT, "data", f"selfdistill{sfx}_train.jsonl")
    tasks = load_tasks()
    if a.langs:
        tasks = [t for t in tasks if t["lang"] in a.langs.split(",")]
    if a.only:
        tasks = [t for t in tasks if any(x in t["id"] for x in a.only.split(","))]
    if a.ids_file:
        keep = {l.strip() for l in open(a.ids_file) if l.strip()}
        tasks = [t for t in tasks if t["id"] in keep]
    if a.limit:
        tasks = random.Random(1).sample(tasks, a.limit)
    print(f"{len(tasks)} tasks: {dict(Counter(t['lang'] for t in tasks))}", flush=True)
    if a.stage == "gen":
        asyncio.run(gen(a, tasks))
    elif a.stage == "score":
        score(a, tasks)
    else:
        build(a, tasks)
