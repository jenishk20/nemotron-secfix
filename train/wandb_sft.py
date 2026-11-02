"""LoRA SFT of Nemotron 3.5 Lightning 30B-A3B on W&B Serverless Training (OpenPipe ART).

Same data as the H200 run (data/secpy_train.jsonl). Training runs on W&B/CoreWeave GPUs
(free during the preview); each checkpoint is hosted for inference right after training.
Needs openpipe-art >= 0.5.23 (Python >= 3.12), so it has its own venv, apart from .venv:
  uv venv .venv-art --python 3.12 && uv pip install --python .venv-art/bin/python "openpipe-art>=0.5.23" "wandb==0.28.0" python-dotenv
and WANDB_API_KEY in .env.

Usage: .venv-art/bin/python train/wandb_sft.py [--name secfix-sft-v1] [--run_name sft-v1] [--epochs 2]
                                               [--data data/secpy_train.jsonl] [--lr 2e-4]
"""
import argparse, asyncio, os

import art
from art.serverless.backend import ServerlessBackend
from art.utils.sft import train_sft_from_file
from dotenv import load_dotenv

ROOT = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(os.path.join(ROOT, ".env"))
BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"


def make_model(name, run_name):
    return art.TrainableModel(name=name, run_name=run_name, project="nemotron-secfix", base_model=BASE)


async def main(name, run_name, data, epochs, lr):
    model = make_model(name, run_name)
    await model.register(ServerlessBackend())  # API key from WANDB_API_KEY; entity filled in by the server
    print("registered:", model.entity, model.project, model.name, "| inference at", model.inference_base_url, flush=True)
    await train_sft_from_file(model=model, file_path=os.path.join(ROOT, data),
                              epochs=epochs, batch_size=2, peak_lr=lr, verbose=True)
    step = await model.get_step()
    print("done:", name, "step", step, "| inference model:", model.get_inference_name(step),
          "| base url:", model.inference_base_url, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="secfix-sft-v1")
    ap.add_argument("--run_name", default="sft-v1")
    ap.add_argument("--data", default="data/secpy_train.jsonl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    asyncio.run(main(a.name, a.run_name, a.data, a.epochs, a.lr))
