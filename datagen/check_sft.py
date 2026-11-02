"""Pre-flight checks for an SFT chat JSONL before paying for a training job.

Checks every row: valid JSON, exactly {"messages": [user, assistant]}, non-empty string contents,
the user turn is CWEval's `direct` prompt around a stub, the assistant turn is one closed code
block (no reasoning leaked), no exact-duplicate rows, and no stub overlapping a CWEval task
(same leakage filter as build_sft.py). With --count_tokens N, measures real token counts for N
random rows through W&B Inference (prompt_tokens of the full chat with max_tokens=1) and
extrapolates the dataset's training tokens and the Fireworks cost.
Usage: .venv-art/bin/python datagen/check_sft.py data/selfdistill_120b_train.jsonl [--count_tokens 60]
"""
import argparse, json, os, random, re, statistics, sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "datagen"))
from build_sft import cweval_prompts, leaks  # noqa: E402

FW_PRICE = 6.00  # $ per 1M LoRA SFT training tokens, Fireworks 80B-300B tier
LANGS = {"python": "py", "py": "py", "c": "c", "cpp": "cpp", "c++": "cpp", "javascript": "js", "js": "js"}


def check(rows):
    errors, stubs, langs, seen = [], [], Counter(), set()
    for i, line in enumerate(rows, 1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"row {i}: invalid JSON ({e})"); continue
        if set(r) != {"messages"}:
            errors.append(f"row {i}: keys {sorted(r)}, expected only 'messages'"); continue
        m = r["messages"]
        if [x.get("role") for x in m] != ["user", "assistant"] or any(set(x) != {"role", "content"} for x in m):
            errors.append(f"row {i}: messages must be exactly [user, assistant] with role/content"); continue
        u, asst = m[0]["content"], m[1]["content"]
        if not isinstance(u, str) or not isinstance(asst, str) or not u.strip() or not asst.strip():
            errors.append(f"row {i}: empty or non-string content"); continue
        if line in seen:
            errors.append(f"row {i}: exact duplicate row")
        seen.add(line)
        if "Your code should be secure" in u:
            errors.append(f"row {i}: user turn still has the security hint (should be the plain direct prompt)")
        stub = re.search(r"```\w*\n(.*)```", u, re.S)
        if not stub:
            errors.append(f"row {i}: no code stub in the user turn"); continue
        stubs.append((i, stub.group(1)))
        a = asst.strip()
        fence = re.match(r"```([\w+]+)\n", a)
        if not fence or not a.endswith("```") or a.count("```") != 2:
            errors.append(f"row {i}: assistant turn is not exactly one closed code block")
        elif fence.group(1) not in LANGS:
            errors.append(f"row {i}: unexpected code fence language {fence.group(1)!r}")
        else:
            langs[LANGS[fence.group(1)]] += 1
        if "<think>" in asst or "</think>" in asst:
            errors.append(f"row {i}: reasoning text leaked into the answer")
    bench = cweval_prompts()
    for i, stub in stubs:
        if (why := leaks(stub, bench)):
            errors.append(f"row {i}: CWEval leak ({why})")
    return errors, langs


def count_tokens(rows, n):
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    client = OpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=os.environ["WANDB_API_KEY"])
    sample = random.Random(0).sample(rows, min(n, len(rows)))
    ratios = []
    for line in sample:
        msgs = json.loads(line)["messages"]
        r = client.chat.completions.create(model="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B", messages=msgs,
                                           max_tokens=1, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        ratios.append(r.usage.prompt_tokens / len(line))
    per_char = statistics.mean(ratios)
    total = per_char * sum(len(l) for l in rows)
    return total, per_char, max(len(l) for l in rows) * max(ratios)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--count_tokens", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    rows = [l.rstrip("\n") for l in open(a.file) if l.strip()]
    errors, langs = check(rows)
    print(f"{a.file}: {len(rows)} rows, by language {dict(langs)}")
    print(f"answer length (chars): median {statistics.median(len(json.loads(l)['messages'][1]['content']) for l in rows):.0f}, "
          f"max {max(len(json.loads(l)['messages'][1]['content']) for l in rows)}")
    print(f"{len(errors)} problems" + (":" if errors else " - format, duplicates, reasoning leaks and CWEval overlap all clean"))
    for e in errors[:30]:
        print("  ", e)
    if a.count_tokens:
        total, per_char, longest = count_tokens(rows, a.count_tokens)
        print(f"tokens (measured on {a.count_tokens} rows via W&B): ~{total / 1e6:.2f}M per epoch, longest row ~{longest:.0f} tokens")
        print(f"Fireworks LoRA SFT cost at ${FW_PRICE}/1M tokens x {a.epochs} epoch(s): ~${total * a.epochs / 1e6 * FW_PRICE:.2f}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
