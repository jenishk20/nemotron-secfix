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

```
Nemotron 3 Ultra 550B (W&B Inference)              Nemotron 3.5 Super 120B (W&B Inference)
  spec + secure ref + vulnerable ref + tests   ──▶   answers each task with reasoning + security hint
  kept only if, in CWEval's Docker image (offline):    each answer run against the task's tests in an
   • secure ref passes all (≥4 func, ≥2 sec tests)       offline W&B Sandbox; keep answers passing ALL
   • vulnerable ref is functional but FAILS ≥1           ──▶ SFT row: plain CWEval prompt → final code
     security test                                      (reasoning removed; ≤2-3 distinct answers/task)
  + up to 2 repair rounds; + CWEval leakage filter
```

**Synthetic task pool:**

| Set | Tasks | Languages | Purpose |
|---|---|---|---|
| Core pool | **1,813** verified | Python 1,009 · C++ 279 · JS 275 · C 250 | 96 CWE × language combinations (`datagen/catalog.py`) |
| v3 targeted | **119** verified (131 before the leakage filter) | Python 43 · C++ 39 · C 19 · JS 18 | CWE-113 header injection, CWE-79 XSS, CWE-918 SSRF (all languages); CWE-347 JWT (C/C++); CWE-327/329 crypto (C++) |

**SFT datasets** (each passed `datagen/check_sft.py`: chat format, no duplicates, no leaked reasoning,
no CWEval overlap):

| Fireworks dataset | Rows | Built from | Used for |
|---|---|---|---|
| `secfix-120b-sd-v1` | 1,073 | 120B, **thinking off** + hint, 4 answers per task, ≤2 kept per task | imported, **never trained** (superseded by thinking-mode data) |
| `secfix-120b-think-v1` | 1,036 | 120B, **thinking on** + hint, 2 answers per task, ≤2 kept | **CyberNemo v1** |
| `secfix-120b-think-v2` | 1,372 | v1 + 2 more thinking answers for each C/C++/JS task, ≤3 kept per task | **CyberNemo v2** |
| `secfix-120b-v3` | 1,132 | v1's 1,036 rows + **96** rows from the v3 targeted tasks | **CyberNemo v3** |
| `secfix-gcs-test`, `upload-test` | 20 each | first 20 rows of v1 | import tests only (direct upload is disabled on our accounts) |

- **Import route:** every dataset was imported from the private bucket `gs://<FW_BUCKET>/<name>.jsonl`
  (`scripts/fireworks_sft.py`), because direct uploads are disabled on our Fireworks accounts.
- **Other datasets:** the `onboarding-*` datasets and the `demo-gsm8k` job are Fireworks' own
  onboarding examples, not part of this project.

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
