"""Score one generation run on W&B Sandboxes (CoreWeave) instead of local Docker.

Same image (co1lin/cweval), same scorer (score_in_container.py), network fully blocked.
One sandbox per language runs in parallel on native x86 (no emulation); each writes
generated_*/res.json for its language, and the merged summary.json is built with the same
summarize() the Docker path uses. Sandboxes show up in W&B under nemotron-secfix, tagged
with the run name and language.

Needs .venv-art (Python 3.12) with "wandb[sandbox]==0.28.0", and WANDB_API_KEY in .env.
Usage: .venv-art/bin/python scripts/score_sandbox.py <run_name> [--cpu 2] [--memory 4Gi]
"""
import argparse, glob, io, json, os, sys, tarfile, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cwsandbox as cw
from dotenv import load_dotenv

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_in_container import summarize  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
ENTITY, PROJECT = "jenishk1800-student", "nemotron-secfix"
CWEVAL = "/home/ubuntu/CWEval"
print_lock = threading.Lock()


def log(*a):
    with print_lock:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def shard_tarball(run_dir, lang):
    """The raw answers for one language (generated_*/<lang>/*_raw.*) plus the scorer, as .tar.gz bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(glob.glob(os.path.join(run_dir, "generated_*", lang, "*_raw.*"))):
            tar.add(p, arcname=os.path.relpath(p, run_dir))
    return buf.getvalue()


def score_shard(run, run_dir, lang, a):
    tag = lang.replace("/", "-")
    t0 = time.time()
    sb = cw.Sandbox.run(
        container_image="co1lin/cweval",
        auth=cw.WandbAuth(entity=ENTITY, project=PROJECT),
        network=cw.NetworkOptions(deny_egress=True, deny_ingress=True),  # never let model code reach the network
        resources={"requests": {"cpu": a.cpu, "memory": a.memory}, "limits": {"cpu": a.cpu, "memory": a.memory}},
        max_lifetime_seconds=a.max_minutes * 60, tags=["secfix-score", run, tag])
    try:
        sb.wait(timeout=900)
        log(f"[{lang}] sandbox up ({time.time() - t0:.0f}s)")
        sb.write_file("/tmp/job.tgz", shard_tarball(run_dir, lang)).result()
        sb.write_file("/tmp/score_in_container.py",
                      open(os.path.join(ROOT, "scripts", "score_in_container.py"), "rb").read()).result()
        cmd = (f". /home/ubuntu/miniforge3/etc/profile.d/conda.sh && cd {CWEVAL} && source .env && "
               f"export GOTOOLCHAIN=local && mkdir -p evals/{run} && tar -xzf /tmp/job.tgz -C evals/{run} && "
               f"python /tmp/score_in_container.py evals/{run} --timeout {a.timeout} --workers {a.workers} && "
               f"cd evals/{run} && tar -czf /tmp/res.tgz generated_*/res.json")
        r = sb.exec(["bash", "-lc", cmd], timeout_seconds=a.max_minutes * 60 - 120).result()
        if r.returncode != 0:
            raise RuntimeError(f"[{lang}] scorer failed rc={r.returncode}: {(r.stdout or '')[-800:]} {(r.stderr or '')[-800:]}")
        data = sb.read_file("/tmp/res.tgz").result()
        log(f"[{lang}] scored ({time.time() - t0:.0f}s)")
        return lang, data
    finally:
        sb.stop().result()


def merge(run_dir, shards):
    """Union the per-language res.json files into run_dir/generated_*/res.json."""
    merged = {}
    for lang, data in shards:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith("res.json"):
                    merged.setdefault(m.name, {}).update(json.load(tar.extractfile(m)))
    for name, res in merged.items():
        with open(os.path.join(run_dir, name), "w") as f:
            json.dump(res, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cpu", default="2")
    ap.add_argument("--memory", default="4Gi")
    ap.add_argument("--timeout", type=int, default=60)   # per test file, same as Docker
    ap.add_argument("--workers", type=int, default=1)    # same as Docker
    ap.add_argument("--max_minutes", type=int, default=60)
    a = ap.parse_args()

    run_dir = os.path.join(ROOT, "outputs", a.run)
    langs = sorted({os.path.join(*p.split(os.sep)[-3:-1])
                    for p in glob.glob(os.path.join(run_dir, "generated_*", "*", "*", "*_raw.*"))})
    assert langs, f"no answers found under {run_dir}"
    log(f"{a.run}: scoring {langs} in {len(langs)} sandboxes")
    for old in glob.glob(os.path.join(run_dir, "generated_*", "res.json")):
        os.remove(old)

    t0, shards, failed = time.time(), [], []
    with ThreadPoolExecutor(len(langs)) as ex:
        futs = {ex.submit(score_shard, a.run, run_dir, lang, a): lang for lang in langs}
        for f in as_completed(futs):
            try:
                shards.append(f.result())
            except Exception as e:
                failed.append(futs[f])
                log(f"[{futs[f]}] FAILED: {type(e).__name__}: {str(e)[:500]}")
    merge(run_dir, shards)
    summarize(run_dir)
    log(f"done in {time.time() - t0:.0f}s; failed shards: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
