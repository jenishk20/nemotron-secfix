"""Short paid eval window on a Fireworks deployment: wait until it is READY, generate all CWEval
answers as fast as possible (scripts/fast_gen.py), then DELETE the deployment, even on failure
or after --max_minutes. Scoring happens afterwards in W&B Sandboxes (no Fireworks cost).
Usage: .venv-art/bin/python scripts/fireworks_eval.py <run> <model_id> [--n 10] [--max_minutes 7]
"""
import argparse, json, os, subprocess, sys, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for line in open(os.path.join(ROOT, ".env")):
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.strip().removeprefix("export ").split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))
ACCOUNT = os.environ["FW_ACCOUNT"]  # your Fireworks account id
API = f"https://api.fireworks.ai/v1/accounts/{ACCOUNT}"


def call(method, path):
    req = urllib.request.Request(f"{API}/{path}", method=method,
                                 headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def find_deployment(model):
    for d in call("GET", "deployments?pageSize=100").get("deployments", []):
        if d.get("baseModel", "").endswith(model.split("/")[-1]) and d.get("state") not in ("DELETED", "DELETING"):
            return d
    return None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("model")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max_minutes", type=float, default=7)  # hard cap on billed time after READY
    ap.add_argument("--thinking_n", type=int, default=0)     # also run <run>_thinkon with reasoning on (n answers/task)
    ap.add_argument("--bench", action="store_true")          # also run the speed benchmark (30 prompts, 4 concurrent)
    ap.add_argument("--wait_minutes", type=float, default=40)
    a = ap.parse_args()
    t0, dep = time.time(), None
    while time.time() - t0 < a.wait_minutes * 60:
        dep = find_deployment(a.model)
        log("deployment:", dep and dep["name"].split("/")[-1], dep and dep.get("state"))
        if dep and dep.get("state") == "READY":
            break
        time.sleep(20)
    if not dep or dep.get("state") != "READY":
        sys.exit("no READY deployment found; nothing generated")
    dep_name = dep["name"].split("/")[-1]
    ready = time.time()
    try:
        target = f"{a.model}#{dep['name']}"  # a LoRA on a dedicated deployment is addressed as model#deployment
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "fast_gen.py"), a.run, f"fireworks:{target}",
               "--n", str(a.n), "--concurrency", str(a.concurrency)]
        steps = [cmd]
        if a.bench:
            steps.append([sys.executable, os.path.join(ROOT, "scripts", "bench_speed.py"), f"{a.run}_speed",
                          f"fireworks:{target}"])
        if a.thinking_n:
            steps.append([sys.executable, os.path.join(ROOT, "scripts", "fast_gen.py"), f"{a.run}_thinkon",
                          f"fireworks:{target}", "--n", str(a.thinking_n), "--concurrency", str(a.concurrency),
                          "--default_mode", "--max_tokens", "16384"])
        for step in steps:  # in priority order; the cap stops whatever is still running
            log("running", os.path.basename(step[1]), step[2])
            p = subprocess.Popen(step)
            while p.poll() is None:
                if time.time() - ready > a.max_minutes * 60:
                    log(f"hit the {a.max_minutes} min cap; stopping"); p.kill(); break
                time.sleep(5)
            if time.time() - ready > a.max_minutes * 60:
                break
    finally:
        # a deployment used in the past hour can only be deleted with ignoreChecks (plain DELETE returns 400)
        call("DELETE", f"deployments/{dep_name}?ignoreChecks=true")
        log(f"deployment {dep_name} DELETED after {(time.time() - ready) / 60:.1f} billed minutes")


if __name__ == "__main__":
    main()
