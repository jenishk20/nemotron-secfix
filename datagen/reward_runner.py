"""Runs INSIDE the CWEval container (offline): score RL rollouts against their task's tests.

<step_dir>/<rollout_id>/ holds impl.py (the model's code), test_task.py and pytest.ini.
Writes <step_dir>/results.json: {rollout_id: {func_pass, func_fail, sec_pass, sec_fail}}.
Same pytest invocation as verify_in_container.py.
Usage: python reward_runner.py <step_dir> [--workers 8]
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_in_container import pytest  # noqa: E402


def score(d):
    f, s = pytest(d, "functionality", timeout=60), pytest(d, "security", timeout=60)
    return os.path.basename(d), {"func_pass": f[1], "func_fail": f[2], "func_timeout": f[0] == "timeout",
                                 "sec_pass": s[1], "sec_fail": s[2], "sec_timeout": s[0] == "timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step_dir")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dirs = sorted(os.path.join(a.step_dir, x) for x in os.listdir(a.step_dir)
                  if os.path.isdir(os.path.join(a.step_dir, x)))
    with ThreadPoolExecutor(a.workers) as ex:
        res = dict(ex.map(score, dirs))
    json.dump(res, open(os.path.join(a.step_dir, "results.json"), "w"))
    print(f"scored {len(res)} rollouts")


if __name__ == "__main__":
    main()
