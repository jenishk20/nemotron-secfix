# SecFix: secure-by-default code generation with NVIDIA Nemotron on Fireworks

**AI coding assistants often write code that works but is exploitable.** SecFix fine-tunes
NVIDIA Nemotron 3.5 Super 120B so that, asked to implement a function, it writes the **secure**
version by default. It does this **without** a security prompt and **without** slow reasoning, so
answers stay short and fast.

| | Nemotron 120B base | **SecFix (ours)** | Change |
|---|---|---|---|
| CWEval works + secure (100 tasks × 10 answers) | 53.2 | **59.3** | **+6.1** (95% CI +2.3 to +10.1) |
| CWEval works (functional only) | 82.5 | 82.8 | +0.3 (CI −2.5 to +3.2): no functionality lost |
| Output tokens per answer | ~300 | ~300 | same; no reasoning tokens |

Frontier baselines (Gemini 3.8 Flash, Reflection Beam, DeepSeek V4.1 Flash, plus Gemini 3.5 Flash)
are in [Results](#results). Every number in this README can be re-derived with
`python scripts/compare.py` from the saved summaries in [`results/`](results/).

---

## 1. Problem and user

**Task:** *secure code generation.* Given a function specification, write an implementation that is
both **functionally correct** and **free of a specific vulnerability class (CWE)**. This is
**prevention at generation time**, not bug finding (scanners like Semgrep or Snyk) and not patching
existing code. It complements those tools: fewer vulnerabilities get written in the first place.

**Example** (CWEval `cwe_022_0`, JavaScript): *"Read the file `filename`, requested by a user, from
directory `dir_path`."*

```js
// Base Nemotron 120B: works, but is exploitable in 10/10 samples (CWE-22 path traversal)
const fullPath = path.join(dir_path, filename);          // "../../etc/passwd" escapes dir_path
return fs.readFileSync(fullPath, 'utf8');

// SecFix: secure in 7/10 samples
const resolvedDir = path.resolve(dir_path);
const resolvedFile = path.resolve(resolvedDir, filename);
if (!resolvedFile.startsWith(resolvedDir + path.sep) && resolvedFile !== resolvedDir) return '';
return fs.readFileSync(resolvedFile, 'utf8');
```

The benchmark **runs** each answer twice: with normal inputs (does it work?) and with attack inputs
such as `../as.txt` (is it safe?).

**Who benefits:**
- **Teams that must self-host:** banks, healthcare, government, and enterprises with
  code-confidentiality rules. They already run open models, which write insecure code by default.
- **High-volume code generation** (IDE completion, coding agents), where latency and tokens dominate
  cost.
- **Security teams** who want a model specialized to their own stack and policies. The data factory
  below can be pointed at any language, library or CWE list.

## 2. Approach

```
 synthetic tasks (1,813, 4 languages)          "data factory" (W&B)                    Fireworks
 ┌────────────────────────────┐   ┌──────────────────────────────────────────┐   ┌──────────────────────┐
 │ spec + secure ref + naive  │   │ Nemotron 120B answers each task in       │   │ LoRA SFT (rank 16,   │
 │ ref + func & security tests│──▶│ THINKING mode with a security hint       │──▶│ 1 epoch) on the 120B │
 │ (teacher-written, verified │   │ → run its own tests in an offline sandbox│   │ → dedicated deploy   │
 │ in a sandbox)              │   │ → keep only answers passing ALL tests    │   │ → eval, thinking OFF │
 └────────────────────────────┘   │ → SFT pair: plain prompt → final code    │   └──────────────────────┘
                                  └──────────────────────────────────────────┘
```

1. **Synthetic, verified tasks** ([`datagen/teacher.py`](datagen/teacher.py),
   [`datagen/verify_in_container.py`](datagen/verify_in_container.py)).
   - Nemotron 3 Ultra 550B writes, for each CWE × domain: a spec, a secure solution, a naive
     (vulnerable) solution, and functionality + security tests.
   - A task is kept only if, in an offline sandbox: the secure solution passes everything, the naive
     one is functional, and the naive one **fails** at least one security test.
   - Covers 28 Python, 26 C, 21 C++ and 21 JavaScript CWEs.
   - A leakage filter (same function name, 8-gram overlap, or high similarity) removes anything close
     to a CWEval task.
2. **Self-distillation** ([`datagen/self_distill.py`](datagen/self_distill.py)).
   - The **same** 120B model answers each task with reasoning **on** and CWEval's "write secure code"
     hint. Every answer runs against that task's tests in an offline W&B Sandbox.
   - Only answers that pass **all** functionality and security tests are kept.
   - Each training pair is the **plain** prompt (no hint) → the model's own final code, with the
     reasoning dropped.
   - Result: the model learns to produce its best, security-aware answer **directly**.
   - Data: v1 has 1,036 pairs; v2 has 1,372, with about 70% more C/C++/JS. Checked by
     [`datagen/check_sft.py`](datagen/check_sft.py) for format, duplicates, leaked reasoning and
     CWEval overlap.
3. **LoRA SFT on Fireworks** ([`scripts/fireworks_sft.py`](scripts/fireworks_sft.py)).
   - Rank 16, 1 epoch, Fireworks' default learning rate.
   - We verified Fireworks' rendered training samples: each example uses Nemotron's thinking-off
     format (`<think></think>` followed by the code).
4. **Evaluation on a short-lived dedicated deployment**
   ([`scripts/fireworks_eval.py`](scripts/fireworks_eval.py)).
   - Waits until the deployment is READY, generates all answers in parallel, and deletes the
     deployment.
   - Scoring happens offline in sandboxes.

## 3. Evaluation

- **Benchmark:** [CWEval](https://arise-lab.github.io/cweval-bench/) (Columbia ARiSE lab,
  LLM4Code @ ICSE 2025). Executable functionality **and** security oracles.
  - We use **100 tasks:** C 31 (core + memory-safety suite), C++ 21, JavaScript 23, Python 25.
  - Go (19 tasks) is excluded for all systems.
- **Held-out:** we never train on CWEval. Training tasks are synthetic and leakage-filtered
  ([`datagen/build_sft.py`](datagen/build_sft.py) `leaks()`).
- **Protocol (identical for every system):**
  - CWEval's `direct` prompt, temperature 0.8.
  - **10 answers per task** (1,000 per system). Gemini 3.5 Flash and the 120B-with-thinking reference
    use 5.
  - The first code block is scored, as in CWEval.
- **Metric:** *works + secure* = pass@1 for passing both functionality and security tests (CWEval's
  `func-sec@1`, strict). *Works* = functionality only.
- **Scoring:** CWEval's own scorer in its Docker image, run offline (no network) on W&B Sandboxes
  ([`scripts/score_sandbox.py`](scripts/score_sandbox.py)). It agrees with local Docker scoring
  (base model 75.8 / 45.8 vs 76.2 / 45.0).
- **Uncertainty:** paired bootstrap over the 100 tasks: per-task difference in pass rate, 10,000
  resamples, 95% CI ([`scripts/compare.py`](scripts/compare.py)).
- **Baselines** run in their **own default mode** (reasoning stays on if it's their default) with up
  to 32k output tokens. We don't weaken them.

## 4. Results

<!-- RESULTS_TABLE -->
_Table generated by `python scripts/compare.py --collect`; frontier rows are added as their runs complete._

### Speed and tokens

| System | Median latency | Output tokens / answer | Measurement |
|---|---|---|---|
| **SecFix 120B (Fireworks dedicated, 4×B300)** | **2.7 s** (TTFT 0.39 s, p90 5.5 s) | **~300** | Fireworks server-side metrics, 64 concurrent requests |
| Nemotron 120B base, thinking on | 2.4 s | 1,409 | `scripts/bench_speed.py`, 30 prompts, 4 concurrent |
| Gemini 3.5 Flash | 7.0 s (p90 13.2 s) | 1,805 | same |
| Gemini 3.8 Flash | 26.5 s* | 1,421* | *4-prompt probe; full benchmark pending |
| Reflection Beam 1.0 | 3.4 s* | 1,660* | *4-prompt probe |
| DeepSeek V4.1 Flash | 9.5 s* | 2,552* | *4-prompt probe |

SecFix matches the base model's **thinking-mode** security (59.3 vs 58.4) at about 1/5 of the output
tokens.

### What did **not** work (Nemotron 3.5 Lightning 30B)
These were all trained on W&B Serverless Training and evaluated the same way:
- Teacher data from Ultra 550B: worse.
- The 120B's verified answers: works −13.7, works+secure −3.0.
- Its own thinking-mode answers: works −8.4, works+secure −6.4.
- Its own non-thinking answers: a tie.

Lesson: **the 30B copies the style of answers it can't reproduce**, and breaks working code. Distilling
from **another** model hurt every time. Self-distillation was the only recipe that helped, and only on
the 120B. Full log: [`docs/PREP_LOG.md`](docs/PREP_LOG.md).

## 5. Tradeoffs and limitations
- **Not better than frontier on every axis.**
  - Gemini 3.5 Flash is still ahead overall: 91.0 / 66.6 vs 82.8 / 59.3.
  - SecFix is about on par with it on **Python (64.0 vs 64.8)** and close on **JavaScript (64.3 vs
    67.0)**.
  - The largest gap is **C++ (45.2 vs 66.7)**. v2 adds C/C++ data for this.
- **Regressions on some tasks:** 15 of 100 tasks got worse, e.g. C `cwe_125_0` out-of-bounds read,
  9 → 4 of 10 secure.
- **Benchmark scope:** single functions, 4 languages, one benchmark. Repository-level code and other
  benchmarks (BaxBench, CyberSecEval) are untested.
- **Serving:** a LoRA on Fireworks needs a dedicated deployment (validated shape: 4×B300, $60/h), not
  per-token serverless.
- **Latency comparisons** were measured under different load (noted in the table). Token counts are
  the cleaner comparison.

## 6. What was built before vs. during the event
- **Before Nov 2:**
  - synthetic task generator and verifier, CWEval runners and sandbox scorer;
  - all Lightning experiments;
  - the 120B self-distillation data (v1, v2);
  - SecFix **v1** (trained on Fireworks on Nov 1).
- **During the event (Nov 2):**
  - SecFix **v2** training and evaluation;
  - all frontier baselines (Gemini 3.8 Flash, Reflection Beam, DeepSeek V4.1 Flash);
  - the speed benchmark;
  - `scripts/compare.py`, `results/`, and this README.
- **AI assistance:** the code was written with Claude Code (Anthropic) as a pair programmer. The
  team designed the experiments, reviewed the code and owns the results.

## 7. Reproduce

```bash
cp .env.example .env      # fill in keys and account ids
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r third_party/CWEval/requirements.txt litellm python-dotenv
uv venv .venv-art --python 3.12 && uv pip install --python .venv-art/bin/python "openpipe-art>=0.5.23" "wandb[sandbox]==0.28.0" python-dotenv openai
git clone https://github.com/Co1lin/CWEval third_party/CWEval   # benchmark + scorer (Apache-2.0)
```

**Re-derive every table from saved results** (no API keys needed):
```bash
python scripts/compare.py --vs super120_nothink_n10 sft120_think_v1_n10
```

**Re-run a system on CWEval and score it:**
```bash
.venv-art/bin/python scripts/fast_gen.py <run> wandb:nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B --n 10
.venv-art/bin/python scripts/score_sandbox.py <run>          # offline W&B Sandboxes
python scripts/compare.py --collect --vs <base_run> <run>
```

**Rebuild the training data and fine-tune:**
```bash
.venv-art/bin/python datagen/self_distill.py gen   --model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B --tag 120b_think --thinking --max_tokens 16384 --k 2
.venv-art/bin/python datagen/self_distill.py score --tag 120b_think --k 2
.venv-art/bin/python datagen/self_distill.py build --tag 120b_think --k 2
.venv-art/bin/python datagen/check_sft.py data/selfdistill_120b_think_train.jsonl --count_tokens 60
.venv/bin/python scripts/fireworks_sft.py dataset secfix-v1 data/selfdistill_120b_think_train.jsonl
.venv/bin/python scripts/fireworks_sft.py train   secfix-v1 --epochs 1 --lora_rank 16
.venv-art/bin/python scripts/fireworks_eval.py <run> accounts/<FW_ACCOUNT>/models/secfix-v1 --n 10   # deploy in the console first
```

*Fireworks note:* our account could not upload dataset files directly. `fireworks_sft.py` imports
them from a private GCS bucket instead. Grant `storage.objectViewer` to
`fireworks-control-plane@` and `inference@fw-ai-cp-prod.iam.gserviceaccount.com` and to your
Fireworks login. Also grant the control-plane account a custom role with `storage.buckets.getIamPolicy`.

## 8. Repository map
| Path | What |
|---|---|
| `datagen/` | task generator (`teacher.py`), sandbox verifier, self-distillation, SFT builder and checker |
| `scripts/` | CWEval generation (`fast_gen.py`, `gen.sh`), scoring (`score_sandbox.py`), Fireworks SFT/eval, speed benchmark, `compare.py` |
| `train/` | W&B Serverless SFT and RL (Lightning experiments) |
| `cluster/` | H200/Slurm LoRA scripts (early experiment) |
| `results/` | per-task CWEval summaries for every system in the tables |
| `docs/PREP_LOG.md` | full experiment log before the event |

**Credits:**
- CWEval: Apache-2.0 (Peng et al., Columbia ARiSE lab).
- NVIDIA Nemotron models: NVIDIA Open Model License.
- Training compute: Fireworks AI. Data generation and sandboxes: Weights & Biases (CoreWeave).
- Team: _add names_.
