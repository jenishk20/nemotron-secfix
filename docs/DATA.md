# Data, models and runs: the complete record

## 1. Test set: CWEval (held out, never trained on)
- **Source:** [CWEval](https://arise-lab.github.io/cweval-bench/) (Columbia ARiSE lab, LLM4Code @ ICSE 2025;
  [GitHub](https://github.com/Co1lin/CWEval), Apache-2.0).
- **Contents:** human-written function specs. Each comes with a reference solution and executable
  **functionality** and **security (attack)** tests.
- **We use 100 of its 119 tasks.** Go (19) is excluded for every system.

| Suite | C (core) | C (memory-safety `lang/c`) | C++ | JavaScript | Python | Total |
|---|---|---|---|---|---|---|
| Tasks | 20 | 11 | 21 | 23 | 25 | **100** |

- **Held out:** every training task passes a leakage filter against all CWEval prompts (same function
  name, any 8-gram overlap, or high similarity), in `datagen/build_sft.py` `leaks()`.
- **Test-informed:** only v3's *choice of CWE families* was informed by CWEval errors.

## 2. Training data: how it is generated

**Short version:**
- Every training answer was written by **Nemotron 3.5 Super 120B itself** (self-distillation).
- **No frontier-model outputs or reasoning traces** (Gemini, DeepSeek, Beam, Claude, ...) are used
  anywhere in training.
- Frontier models were only *evaluated* on the test set.

### Who produced what

| Part of a training example | Produced by | Where | Kept? |
|---|---|---|---|
| Task spec, secure + vulnerable reference solutions, functionality + attack tests | **Nemotron 3 Ultra 550B** (`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`), the "teacher" | W&B Inference | Only to **define tasks**; its code is never a training answer |
| Check that each task's tests actually catch the vulnerability | CWEval's Docker image, offline | local Docker / W&B Sandboxes | Task kept only if it passes |
| **The training answer** (the code the model learns to write) | **Nemotron 3.5 Super 120B**, the same model we fine-tune, with reasoning on + a security hint | W&B Inference | Only if it passes **all** the task's tests in an offline sandbox |
| Its **reasoning trace** | Nemotron 120B | — | **Dropped.** We train on the final code only, with the plain prompt (no hint) |

### Step by step
1. **Tasks** (`datagen/teacher.py`).
   - The 550B teacher writes, for a given CWE and application domain: a function spec, a secure
     solution, a deliberately vulnerable solution, and pytest tests (≥4 functionality, ≥2 security).
2. **Task verification** (`datagen/verify_in_container.py`), offline.
   - A task is kept only if the secure solution passes every test, the vulnerable one passes the
     functionality tests **and fails ≥1 security test**.
   - Failures get up to 2 repair rounds with the error log.
   - The CWEval leakage filter is applied afterwards.
3. **Answers** (`datagen/self_distill.py`).
   - The 120B answers each task with CWEval's prompt + "Your code should be secure…", **thinking on**.
   - Each answer runs against that task's own tests in an offline W&B Sandbox.
4. **Training rows.**
   - Only answers passing **all** tests are kept, at most 2–3 distinct answers per task.
   - Each row is `{user: plain CWEval prompt, assistant: final code block}`.

### The three training datasets (all on Fireworks, imported from a private GCS bucket)

| Dataset | Rows | Tasks it draws on | Answers sampled per task (120B, thinking on) | Rows kept per task | Trained model |
|---|---|---|---|---|---|
| `secfix-120b-think-v1` | **1,036** | 1,813 core tasks (Py 1,009 · C++ 279 · JS 275 · C 250) | 2 | ≤2 | **CyberNemo v1** |
| `secfix-120b-think-v2` | **1,372** | the same 1,813 tasks | 2 (Python) / **4** (C, C++, JS) | ≤3 | **CyberNemo v2** |
| `secfix-120b-v3` | **1,132** | v1's rows + **119 new targeted tasks** (CWE-113 header injection, CWE-79 XSS, CWE-918 SSRF, CWE-347 JWT, CWE-327/329 crypto) | 4 on the new tasks | ≤3 | **CyberNemo v3** |

- **v2 vs v1:** same tasks; more answers sampled for C/C++/JS, where v1 was weakest. That gives
  ~70% more C/C++/JS rows.
- **v3 vs v1:** new tasks for the CWE families found in the error analysis.
  - Only 96 new rows survived: the 120B rarely passes these tasks even with reasoning.
  - Only the *choice of families* was informed by CWEval; no CWEval content is used.
- **`secfix-120b-sd-v1`** (1,073 rows, 120B with thinking **off**) was imported to Fireworks but never
  trained. The thinking-mode data replaced it.

## 3. Models trained on Fireworks
Base: `accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16`. LoRA rank 16, 1 epoch,
Fireworks' default learning rate (1e-4, constant).

| Model | Account | Dataset | Steps | Training cost |
|---|---|---|---|---|
| `secfix-120b-think-v1` (CyberNemo v1) | first account | `secfix-120b-think-v1` | 33 | $8.19 |
| `secfix-120b-think-v2` (CyberNemo v2) | event account | `secfix-120b-think-v2` | 43 | $11.15 |
| `secfix-120b-v3` (CyberNemo v3) | event account | `secfix-120b-v3` | ~36 | $9.16 |

**Evaluation deployments:**
- Each was a dedicated deployment (4× B300, BF16), created only for an eval window and auto-deleted by
  `scripts/fireworks_eval.py`.
- Today's windows: 3.5 min, 16.0 min and 4.9 min.

## 4. Results on the test set (works + secure, CWEval pass@1)

| System | Mode | Answers/task | Works + secure | Works | Summary file |
|---|---|---|---|---|---|
| Nemotron 120B base | thinking off | 10 | 53.2 | 82.5 | `results/super120_nothink_n10` |
| Nemotron 120B base | thinking on | 5 | 58.4 | 86.4 | `results/base_wandb_all_n5` |
| CyberNemo v1 | thinking off | 10 | 59.3 | 82.8 | `results/sft120_think_v1_n10` |
| CyberNemo v2 | thinking off | 10 | 57.6 | 81.4 | `results/sft120_think_v2_n10` |
| CyberNemo v2 | thinking on | 10 | 63.3 | 86.2 | `results/sft120_think_v2_n10_thinkon` |
| **CyberNemo v2 + security review** | thinking on, 2 calls | 9 (869 answers) | **72.4** | 81.1 | `results/sft120_think_v2_n10_thinkon_review` |
| CyberNemo v3 | thinking on | 10 | 64.6 | 85.6 | `results/sftv3_thinkon` |
| Reflection Beam 1.0 | default | 10 | 76.2 | 86.9 | `results/beam_n10` |
| Gemini 3.8 Flash | default | 10 | 74.3 | 88.9 | `results/gemini38flash_n10` |
| DeepSeek V4.1 Flash | default | 10 | 71.3 | 89.0 | `results/dsv41flash_n10` |
| Gemini 3.5 Flash | default | 5 | 66.6 | 91.0 | `results/gemini35flash_all_n5` |

**Reproduce:** `python scripts/compare.py --vs <base_run> <run>` prints the table and paired 95% CIs.
