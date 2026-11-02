"""Runs INSIDE the CWEval container (offline): score RL rollouts against their task's tests.

<step_dir>/<rollout_id>/ holds impl.py (the model's code), test_task.py and pytest.ini; for C/C++/JS,
code.txt (the model's code) and meta.json ({lang, entrypoint}) instead of impl.py, and the program is
built exactly as verify_in_container.py builds the teacher's (CWEval headers + entrypoint, ASan).
Writes <step_dir>/results.json: {rollout_id: {func_pass, func_fail, sec_pass, sec_fail, compile_error}}.
Same pytest invocation as verify_in_container.py.
Usage: python reward_runner.py <step_dir> [--workers 8]
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_in_container import HARNESS, build, impl_env, pytest  # noqa: E402


def score(d):
    env = None
    if os.path.exists(os.path.join(d, "meta.json")):  # C/C++/JS: write the run_impl harness, build the program
        m = json.load(open(os.path.join(d, "meta.json")))
        open(os.path.join(d, "harness.py"), "w").write(HARNESS)
        rec = {"lang": m["lang"], "entrypoint": m["entrypoint"], "code": open(os.path.join(d, "code.txt")).read()}
        try:
            ok, _ = build(d, rec, "code")
        except Exception:
            ok = False
        if not ok:
            return os.path.basename(d), {"func_pass": 0, "func_fail": 1, "func_timeout": False,
                                         "sec_pass": 0, "sec_fail": 1, "sec_timeout": False, "compile_error": True}
        env = impl_env(d, m["lang"])
    f, s = pytest(d, "functionality", timeout=60, env=env), pytest(d, "security", timeout=60, env=env)
    return os.path.basename(d), {"func_pass": f[1], "func_fail": f[2], "func_timeout": f[0] == "timeout",
                                 "sec_pass": s[1], "sec_fail": s[2], "sec_timeout": s[0] == "timeout",
                                 "compile_error": False}


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
