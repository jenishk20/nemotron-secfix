"""LoRA SFT of Nemotron 3.5 Lightning 30B-A3B on W&B Serverless Training (OpenPipe ART).

Same data as the H200 run (data/secpy_train.jsonl). Training runs on W&B/CoreWeave GPUs
(free during the preview); each checkpoint is hosted for inference right after training.
Needs openpipe-art >= 0.5.23 (Python >= 3.12), so it has its own venv, apart from .venv:
  uv venv .venv-art --python 3.12 && uv pip install --python .venv-art/bin/python "openpipe-art>=0.5.23" python-dotenv
and WANDB_API_KEY in .env.

Usage: .venv-art/bin/python train/wandb_sft.py [--name secfix-sft-v1] [--epochs 2]
"""
import argparse, asyncio, os

import art
from art.serverless.backend import ServerlessBackend
from art.utils.sft import train_sft_from_file
from dotenv import load_dotenv

ROOT = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(os.path.join(ROOT, ".env"))
BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"


async def main(name, epochs, lr):
    model = art.TrainableModel(name=name, project="nemotron-secfix", base_model=BASE)
    await model.register(ServerlessBackend())
    await train_sft_from_file(model=model, file_path=os.path.join(ROOT, "data/secpy_train.jsonl"),
                              epochs=epochs, batch_size=2, peak_lr=lr)
    print("done:", name, "step", await model.get_step())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="secfix-sft-v1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    asyncio.run(main(a.name, a.epochs, a.lr))
