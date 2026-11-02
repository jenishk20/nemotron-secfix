"""RL (GRPO-style) for secure Python code on W&B Serverless Training (OpenPipe ART).

Continues a W&B model (default: the SFT model secfix-sft-v1; a new name starts from the base
Nemotron 3.5 Lightning 30B-A3B). Each step: sample tasks from our verified synthetic Python tasks
(never CWEval), ask the model for --group answers per task with exactly the CWEval `direct` prompt
(thinking off, W&B Inference), run each answer against that task's own tests in an offline
W&B Sandbox (datagen/reward_runner.py), and train on the rewards (W&B Serverless Training):

  reward = 1.0  if every functionality and security test passes   (what CWEval counts)
         = 0.3  if every functionality test passes but a security test fails
         = 0.2 * fraction of functionality tests passed, otherwise   (0 on timeout)

Needs .venv-art (openpipe-art, "wandb[sandbox]", python-dotenv) and WANDB_API_KEY in .env.
Usage: .venv-art/bin/python train/wandb_rl.py [--name secfix-sft-v1 --run_name sft-v1] [--steps 30] [--tasks 16] [--group 8]
Each checkpoint is hosted on W&B; evaluate with scripts/gen.sh wbtrain:<inference name>.
"""
import argparse, asyncio, glob, io, json, os, random, re, sys, tarfile, time

import art
import cwsandbox as cw
from art.serverless.backend import ServerlessBackend
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "datagen"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "CWEval"))
from build_sft import cweval_prompts, leaks  # noqa: E402
from cweval.ppt import DirectPrompt  # noqa: E402
from verify_in_container import PYTEST_INI  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"
ENTITY, PROJECT = os.environ["WANDB_ENTITY"], os.environ.get("WANDB_PROJECT", "nemotron-secfix")
ENV = ". /home/ubuntu/miniforge3/etc/profile.d/conda.sh && cd /home/ubuntu/CWEval && source .env"


def load_tasks(holdout, seed):
    """Verified Python tasks (any repair round), minus anything overlapping CWEval."""
    recs, bench = {}, cweval_prompts()
    for f in sorted(glob.glob(os.path.join(ROOT, "data/synth/py_*.jsonl"))):
        if f.endswith(".verify.jsonl") or not os.path.exists(f.replace(".jsonl", ".verify.jsonl")):
            continue
        ok = {r["id"] for r in map(json.loads, open(f.replace(".jsonl", ".verify.jsonl"))) if r.get("ok")}
        for r in map(json.loads, open(f)):
            if r["id"] in ok and r["id"] not in recs and not leaks(r["stub"], bench):
                recs[r["id"]] = r
    tasks = sorted(recs.values(), key=lambda r: r["id"])
    random.Random(seed).shuffle(tasks)
    return tasks[holdout:], tasks[:holdout]


def prompt(task):
    return DirectPrompt.PPT.format(lang="py", lang_instr=DirectPrompt.LANG_INSTR["py"],
                                   code_prompt=task["stub"].rstrip("\n"))


def first_code_block(text):
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text or "", re.S)
    return m.group(1) if m else (text or "")


def start_sandbox(cpu, memory, hours, tags):
    """One offline sandbox (CWEval image) for the whole run; the reward scripts are uploaded once."""
    sb = cw.Sandbox.run(container_image="co1lin/cweval", auth=cw.WandbAuth(entity=ENTITY, project=PROJECT),
                        network=cw.NetworkOptions(deny_egress=True, deny_ingress=True),  # model code stays offline
                        resources={"requests": {"cpu": cpu, "memory": memory}, "limits": {"cpu": cpu, "memory": memory}},
                        max_lifetime_seconds=int(hours * 3600), tags=["secfix-rl"] + tags)
    sb.wait(timeout=900)
    sb.exec(["mkdir", "-p", "/tmp/datagen", "/tmp/work"]).result()
    for f in ("verify_in_container.py", "reward_runner.py"):
        sb.write_file(f"/tmp/datagen/{f}", open(os.path.join(ROOT, "datagen", f), "rb").read()).result()
    return sb


def score_rollouts(sb, step, items, workers):
    """items: [(rollout_id, task, code)] -> {rollout_id: test counts}, run in the offline sandbox."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rid, task, code in items:
            for name, text in (("impl.py", code or ""), ("test_task.py", task["tests"]), ("pytest.ini", PYTEST_INI)):
                data = text.encode()
                info = tarfile.TarInfo(f"step_{step}/{rid}/{name}"); info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    sb.write_file(f"/tmp/step_{step}.tgz", buf.getvalue()).result()
    cmd = (f"{ENV} && tar -xzf /tmp/step_{step}.tgz -C /tmp/work && "
           f"python /tmp/datagen/reward_runner.py /tmp/work/step_{step} --workers {workers}")
    r = sb.exec(["bash", "-lc", cmd], timeout_seconds=1800).result()
    if r.returncode != 0:
        raise RuntimeError(f"reward runner failed (rc={r.returncode}): {((r.stdout or '') + (r.stderr or ''))[-2000:]}")
    res = json.loads(sb.read_file(f"/tmp/work/step_{step}/results.json").result())
    sb.exec(["rm", "-rf", f"/tmp/work/step_{step}", f"/tmp/step_{step}.tgz"]).result()
    return res


def reward(c):
    if c["func_timeout"] or c["sec_timeout"]:
        return 0.0
    f_tot = c["func_pass"] + c["func_fail"]
    if not f_tot or c["func_fail"]:
        return 0.2 * (c["func_pass"] / f_tot if f_tot else 0.0)
    return 1.0 if c["sec_fail"] == 0 and c["sec_pass"] > 0 else 0.3


async def complete(client, sem, **kw):
    """One completion, at most `sem` in flight; retries rate limits (429) and server errors with backoff."""
    for attempt in range(8):
        try:
            async with sem:
                return await client.chat.completions.create(**kw)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if attempt == 7 or not (status == 429 or (status and status >= 500) or "timeout" in type(e).__name__.lower()):
                raise
            await asyncio.sleep(min(60, 2 ** attempt) + random.random())


async def rollout_group(model, client, sem, task, group, max_tokens, step):
    messages = [{"role": "user", "content": prompt(task)}]
    # W&B serves LoRA checkpoints only by explicit step; a fresh model (step 0) samples from the base
    model_id = model.get_inference_name(step) if step > 0 else BASE
    comps = await asyncio.gather(*[complete(client, sem, model=model_id, messages=messages, temperature=1.0,
                                            max_tokens=max_tokens) for _ in range(group)],
                                 return_exceptions=True)
    choices = [c.choices[0] for c in comps if not isinstance(c, BaseException)]
    errors = [c for c in comps if isinstance(c, BaseException)]
    if errors:
        print(f"  {len(errors)} of {group} rollouts failed: {type(errors[0]).__name__}: {str(errors[0])[:160]}", flush=True)
    return task, messages, choices


async def main(a):
    train_tasks, holdout = load_tasks(a.holdout, a.seed)
    print(f"{len(train_tasks)} training tasks, {len(holdout)} held out", flush=True)
    model = art.TrainableModel(name=a.name, run_name=a.run_name, project="nemotron-secfix", base_model=BASE,
                               _internal_config={"chat_template_kwargs": {"enable_thinking": False}})
    backend = ServerlessBackend()
    await model.register(backend)
    client = model.openai_client()
    first = await model.get_step()
    print(f"model {a.name} at step {first}; training {a.steps} RL steps", flush=True)
    sb = start_sandbox(a.cpu, a.memory, a.hours, [a.name])
    print(f"reward sandbox up: {sb.sandbox_id if hasattr(sb, 'sandbox_id') else sb}", flush=True)
    rng = random.Random(a.seed + first)
    sem = asyncio.Semaphore(a.concurrency)
    try:
        for step in range(first, first + a.steps):
            t0 = time.time()
            batch = rng.sample(train_tasks, a.tasks)
            outs = await asyncio.gather(*[rollout_group(model, client, sem, t, a.group, a.max_tokens, step) for t in batch],
                                        return_exceptions=True)
            items, meta = [], {}
            for gi, out in enumerate(outs):
                if isinstance(out, BaseException):
                    print(f"  rollout error: {type(out).__name__}: {str(out)[:200]}", flush=True)
                    continue
                task, messages, choices = out
                for ci, ch in enumerate(choices):
                    rid = f"g{gi}_c{ci}"
                    items.append((rid, task, first_code_block(ch.message.content)))
                    meta[rid] = (gi, messages, ch)
            t1 = time.time()
            if len({gi for gi, _, _ in meta.values()}) < 2:
                print(f"step {step}: too few rollouts came back ({len(items)}); skipping this step", flush=True)
                continue
            counts = await asyncio.to_thread(score_rollouts, sb, step, items, a.workers)
            groups = {}
            for rid, (gi, messages, ch) in meta.items():
                c = counts.get(rid, {"func_pass": 0, "func_fail": 1, "func_timeout": True,
                                     "sec_pass": 0, "sec_fail": 1, "sec_timeout": True})
                r = reward(c)
                works = c["func_fail"] == 0 and c["func_pass"] > 0 and not c["func_timeout"]
                secure = works and c["sec_fail"] == 0 and c["sec_pass"] > 0 and not c["sec_timeout"]
                groups.setdefault(gi, []).append(art.Trajectory(
                    messages_and_choices=[*messages, ch], reward=r,
                    metrics={"works": works, "works_and_secure": secure}))
            tgroups = [art.TrajectoryGroup(ts) for ts in groups.values() if len(ts) >= 2]  # GRPO needs a group
            rewards = [t.reward for g in tgroups for t in g.trajectories]
            ws = sum(t.metrics["works_and_secure"] for g in tgroups for t in g.trajectories) / max(1, len(rewards))
            print(f"step {step}: {len(rewards)} rollouts, mean reward {sum(rewards) / max(1, len(rewards)):.3f}, "
                  f"works+secure {ws:.1%} | rollouts {t1 - t0:.0f}s, tests {time.time() - t1:.0f}s", flush=True)
            await model.log(tgroups, split="train")
            await backend.train(model, tgroups, learning_rate=a.lr)
            print(f"  trained -> step {await model.get_step()} ({time.time() - t0:.0f}s total); "
                  f"inference model: {model.get_inference_name(await model.get_step())}", flush=True)
    finally:
        sb.stop().result()
        print("reward sandbox stopped", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="secfix-sft-v1")   # continue the SFT model
    ap.add_argument("--run_name", default="sft-v1")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--tasks", type=int, default=16)   # tasks per step
    ap.add_argument("--group", type=int, default=8)    # answers per task (GRPO group)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--holdout", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", default="8")
    ap.add_argument("--memory", default="16Gi")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hours", type=float, default=3)  # sandbox max lifetime (cost cap)
    ap.add_argument("--concurrency", type=int, default=16)  # W&B Inference rejects too many parallel requests
    asyncio.run(main(ap.parse_args()))
