"""One-call smoke test for Nemotron on W&B Inference.

Answers the three things that decide whether CWEval's own generate.py can
talk to W&B directly:
  1. Does auth work (and is a team/project header required)?
  2. Does the endpoint accept n>1 in one request? (CWEval sends n in one call
     for any `openai/` model; if this fails, we pass --n 1 per run instead.)
  3. Does the answer land in `content`, or does reasoning eat the token budget?

Usage:
  .venv/bin/python scripts/smoke_wandb.py
Optional env (in .env): WANDB_PROJECT_PATH=<team>/<project>
"""

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    base_url=config.WANDB_INFERENCE_BASE_URL,
    api_key=os.environ["WANDB_API_KEY"],
    project=os.environ.get("WANDB_PROJECT_PATH"),  # sent as OpenAI-Project header
)

PROMPT = "Write a Python function `mean(xs)` that returns the average of a list. Code only."


def call(n: int, max_tokens: int):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=config.WANDB_NEMOTRON,
        messages=[{"role": "user", "content": PROMPT}],
        n=n,
        max_tokens=max_tokens,
        temperature=0.8,
    )
    dt = time.time() - t0
    print(f"\n=== n={n} max_tokens={max_tokens} -> {len(resp.choices)} choices in {dt:.1f}s")
    print(f"usage: {resp.usage}")
    for i, c in enumerate(resp.choices):
        msg = c.message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        print(f"[{i}] finish={c.finish_reason} content_chars={len(msg.content or '')} "
              f"reasoning_chars={len(reasoning or '')}")
        print((msg.content or "<EMPTY CONTENT>")[:400])


if __name__ == "__main__":
    call(n=1, max_tokens=2048)   # same budget CWEval uses by default
    try:
        call(n=2, max_tokens=2048)
    except Exception as e:
        print(f"\n=== n=2 FAILED -> run CWEval with --n 1 per pass. Error: {e}")
