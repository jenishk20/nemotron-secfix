---
license: other
license_name: research-use
pretty_name: Verified synthetic secure-coding tasks
tags: [code, security, secure-code-generation, synthetic, sft]
configs:
- config_name: python
  data_files: tasks/python.jsonl
- config_name: c
  data_files: tasks/c.jsonl
- config_name: cpp
  data_files: tasks/cpp.jsonl
- config_name: javascript
  data_files: tasks/javascript.jsonl
- config_name: sft_python_v1
  data_files:
  - split: train
    path: sft/python_v1_train.jsonl
  - split: validation
    path: sft/python_v1_val.jsonl
---

# Verified synthetic secure-coding tasks

Training data for teaching an open model (NVIDIA Nemotron) to write code that **works and is
secure**. Every task is new, written in the same shape as the CWEval benchmark, and **executed
before it was kept**. For defensive use: secure code generation and fixing, not exploitation.

## How each task was made

1. A teacher model (**NVIDIA Nemotron 3 Ultra 550B**, via W&B Inference) writes a task for one
   CWE and one application area: a function **prompt** (signature + doc comment that never names
   the weakness), a **secure solution**, an **insecure solution** a hurried developer might
   write, and **pytest tests** (functionality + security). C/C++/JS tasks also get an
   **entrypoint** (`main` / `require.main`) that reads command-line arguments, like CWEval's.
2. The tests run in an offline sandbox (the CWEval Docker image). A task is kept only if
   - the secure solution passes **every** test (at least 4 functionality and 2 security tests),
   - the insecure solution passes the functionality tests, and
   - the insecure solution **fails** at least one security test (so the tests really detect it).
3. Failed tasks go back to the teacher with the test log, up to 2 repair rounds.
4. Anything resembling a CWEval task (same function name, a shared 8-word phrase, or
   near-duplicate text) is removed, so training never sees the benchmark.

## Configs

- `python`, `c`, `cpp`, `javascript`: one row per verified task (see fields below).
  Python is the main set (~1,000 tasks over 28 CWEs); C/C++/JS are early pilots.
- `sft_python_v1`: the chat-format rows used for our first fine-tune
  (`{"messages": [user, assistant]}`; the user turn is exactly CWEval's `direct` prompt,
  the assistant turn is the verified secure solution).

| field | meaning |
|---|---|
| `task_id`, `language`, `cwe`, `cwe_description`, `domain` | what the task is about |
| `prompt` | what the student model sees |
| `secure_solution` | verified secure answer (the training target) |
| `insecure_solution` | **intentionally vulnerable** answer, used only to prove the tests detect the weakness |
| `entrypoint` | C/C++/JS only: the `main` that runs the function from command-line arguments |
| `tests` | pytest file; C/C++/JS tests call the program through `harness.run_impl(*args)` |
| `functionality_tests`, `security_tests` | how many tests of each kind the secure solution passed |
| `repair_rounds` | 0 = passed first time, 1-2 = passed after the teacher fixed it |
| `teacher` | model that wrote the task |

**Warning:** `insecure_solution` fields and some test inputs are deliberately vulnerable or
hostile. Never deploy them. Generated with NVIDIA Nemotron 3 Ultra; check that model's license
terms before any non-research use.
