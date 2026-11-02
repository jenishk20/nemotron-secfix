"""LoRA SFT of Nemotron 3.5 Super VL 120B on Fireworks, with the dataset imported from GCS.

Our Fireworks account can't upload files ("dataset uploads are temporarily disabled"), but it can
import from a private GCS bucket that grants read access to Fireworks' service accounts
(fireworks-control-plane@ and inference@fw-ai-cp-prod, see README). Steps:
  dataset <name> <file.jsonl>   copy the file to gs://$FW_BUCKET/<name>.jsonl, create the dataset, wait for READY
  train   <name> [--epochs 1 --lora_rank 16 --lr X]   start the SFT job; output model = <name>
  status  <name>                job state
Needs FIREWORKS_API_KEY in .env and gcloud signed in. Usage: .venv/bin/python scripts/fireworks_sft.py ...
"""
import argparse, json, os, subprocess, time, urllib.error, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for line in open(os.path.join(ROOT, ".env")):
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.strip().removeprefix("export ").split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))
API = "https://api.fireworks.ai/v1"
ACCOUNT = os.environ["FW_ACCOUNT"]  # your Fireworks account id
BUCKET = os.environ["FW_BUCKET"]  # private GCS bucket readable by Fireworks (see README)
BASE = "accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16"


def call(method, path, body=None):
    req = urllib.request.Request(f"{API}/accounts/{ACCOUNT}/{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
                                          "Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:800]}")


def dataset(a):
    n = sum(1 for _ in open(a.file))
    subprocess.run(["gsutil", "-q", "cp", a.file, f"gs://{BUCKET}/{a.name}.jsonl"], check=True)
    call("POST", "datasets", {"datasetId": a.name, "dataset": {
        "externalUrl": f"gs://{BUCKET}/{a.name}.jsonl", "format": "CHAT", "exampleCount": str(n)}})
    for _ in range(60):
        d = call("GET", f"datasets/{a.name}")
        print("dataset", a.name, d.get("state"), d.get("status", {}).get("message", ""), flush=True)
        if d.get("state") == "READY":
            return
        time.sleep(10)


def train(a):
    body = {"dataset": f"accounts/{ACCOUNT}/datasets/{a.name}", "baseModel": BASE,
            "outputModel": f"accounts/{ACCOUNT}/models/{a.name}", "displayName": a.name,
            "epochs": a.epochs, "loraRank": a.lora_rank}  # loraRank must be set: 0 means full-parameter training
    if a.lr:
        body["learningRate"] = a.lr
    j = call("POST", f"supervisedFineTuningJobs?supervisedFineTuningJobId={a.name}", body)
    print(json.dumps({k: j.get(k) for k in ("name", "state", "baseModel", "outputModel", "epochs",
                                            "loraRank", "learningRate")}, indent=1))


def status(a):
    j = call("GET", f"supervisedFineTuningJobs/{a.name}")
    print(json.dumps({k: j.get(k) for k in ("name", "state", "status", "outputModel", "createTime",
                                            "updateTime", "jobProgress")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dataset", "train", "status"])
    ap.add_argument("name")
    ap.add_argument("file", nargs="?")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0)  # 0 = Fireworks' model default
    a = ap.parse_args()
    {"dataset": dataset, "train": train, "status": status}[a.cmd](a)
