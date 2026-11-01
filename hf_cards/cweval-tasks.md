---
license: apache-2.0
pretty_name: CWEval tasks (JSONL mirror) with model results
tags: [code, security, secure-code-generation, benchmark, cwe]
configs:
- config_name: tasks
  data_files: data/tasks.jsonl
- config_name: results
  data_files: results/results.jsonl
---

# CWEval tasks, with our model results

A browsable copy of the **CWEval** benchmark plus the per-task results from our
secure-code project. CWEval asks a model to complete a function from its signature and doc
comment, then runs the code: **functionality tests** (does it work?) and **security tests**
(hostile inputs: path traversal, injection, weak crypto, memory errors, ...). The headline
metric is **works AND secure** (`func-sec@1`).

**Original work:** CWEval by Jinjun Peng et al., <https://github.com/Co1lin/CWEval>
(paper: arXiv 2501.08200). Licensed Apache-2.0; the license is included here.
**Our changes:** converted the benchmark files to JSONL (one row per task) and added a results
table. Task content is unchanged. Please cite the original paper if you use this.

## `tasks` (119 rows)

| field | meaning |
|---|---|
| `task_id` | e.g. `core/py/cwe_022_0` (suite / language / CWE / index) |
| `suite` | `core` (cross-language CWEs) or `lang` (C-specific memory safety) |
| `language`, `cwe` | language and CWE number |
| `prompt` | exactly what the model is shown (doc comment + signature) |
| `reference_solution` | the benchmark's secure reference, full file |
| `insecure_reference` | the benchmark's intentionally vulnerable reference (C/C++/JS/Go; for Python it lives in the test file) |
| `tests` | the pytest file with functionality and security tests |

## `results`

One row per (task, model): `samples` answers were generated (n=5, temperature 0.8, CWEval's
`direct` prompt), and `works` / `works_and_secure` count how many passed. Scored strictly
(an answer with no code counts as a failure) in the offline CWEval Docker image.

Overall (100 tasks, Go excluded because its tests need network access):

| model | works | works + secure |
|---|---|---|
| Nemotron 3.5 Super VL 120B | 86.4 | 58.4 |
| Nemotron 3.5 Lightning 30B-A3B, thinking off | 76.2 | 45.0 |
| Nemotron 3.5 Lightning 30B-A3B, thinking on | 76.6 | 50.6 |
| Lightning 30B-A3B + SFT v1 (Python-only data) | 49.4 | 38.2 |
| Gemini 3.5 Flash | 91.0 | 66.6 |
| Muse Glimmer 30B | 85.2 | 60.4 |
| Claude Sonnet 5.5 (Python tasks only) | 96.0 | 87.2 |

**Warning:** `insecure_reference` and some test inputs are deliberately vulnerable or hostile.
They exist to check defenses; do not deploy that code.
