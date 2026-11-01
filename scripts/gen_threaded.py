"""CWEval generation with threads instead of forked processes.

CWEval's generate.py runs model calls in forked worker processes (p_map). On macOS the
OpenRouter client deadlocks after fork: the first call in a worker succeeds, later ones
hang forever with no error. This reuses CWEval's own Gener (same prompts, same output
layout, same skip-if-done resume) but runs `_gen_case` in a thread pool, and adds a hard
per-request timeout and retries.

Run from third_party/CWEval with PYTHONPATH set (scripts/gen.sh does this when GEN_THREADS=1).
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
from tqdm import tqdm

from cweval.generate import Gener


def gen_with_retry(*args, tries: int = 4):
    # litellm's own retry needs `tenacity`; retry here so transient provider errors don't drop tasks
    for attempt in range(tries):
        try:
            return Gener._gen_case(*args)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def main(threads: int = 6, request_timeout: int = 300, **kwargs):
    g = Gener(timeout=request_timeout, **kwargs)  # extra kwargs flow into litellm.completion
    failed = []
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(gen_with_retry, g.model, g.ppt, case, g.ai_kwargs, i): path
                for i, (path, case) in enumerate(g.cases.items())}
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                f.result()
            except Exception as e:  # leave the task unwritten; a re-run fills it in
                failed.append(futs[f])
                print(f"FAILED {futs[f]}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print(f"done: {len(g.cases) - len(failed)}/{len(g.cases)} tasks, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    fire.Fire(main)
