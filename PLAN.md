# PLAN.md: TrialGPT Criterion-Matching System (Generative AI Assignment #2)

You are an AI coding agent (opencode + DeepSeek). Build this project end to end, phase by phase, **autonomously**, ticking every checklist item. The user does NOT review individual phases; you verify your own work. Deadline: **2026-11-10** (2h grace). Effort budget: about 6 days.

**Context hygiene (important for your limited context):** do not re-read this whole file every time. At session start read sections A, 0, and 1, then only the section of the phase you are working on (find it with grep on `^## `). Keep `PROGRESS.md` short.

---

## A. START HERE: session protocol (any agent, any session)

This project may be handed between agents/models mid-way (token limits, tool switches). **The repo itself is the memory.** Follow this protocol every session so the next agent can continue without the user explaining anything.

### A.1 On session start (always, before touching code)
1. Read `PLAN.md` fully (this file), then `PROGRESS.md`, then `docs/DECISIONS.md` if it exists.
2. Run `git log --oneline -20` and `git status`. If the working tree is dirty, finish or stash that work before anything else, and note it in `PROGRESS.md`.
3. Run the quality gate (section 3.2). If it is red, fixing it is the first task.
4. Resume at the item in `PROGRESS.md -> "NEXT ACTION"`. Do not redo completed items; do not skip ahead to a later phase.

### A.2 During work (autonomous mode)
- Work through the phases **in order, without waiting for approval**. At the end of each phase: run the **self-review** (A.2.1), quality gate green, `PROGRESS.md` updated, `git commit` (one commit per phase, message `phase N: <summary>`), `git tag phase-N-done`, then **immediately start the next phase**.
- Update `PROGRESS.md` **after every meaningful step** (a module finished, a test passing, a decision made), not only at phase end. Assume you may be cut off at any moment.
- Make small commits on the way (`wip: <what>`); never leave the repo uncommittable.
- Record every decision that is not already in this plan in `docs/DECISIONS.md` (what, why, alternatives). When something is ambiguous, pick the simplest option consistent with the assignment and log it; do not ask the user.
- **The only reasons to stop and talk to the user** are the two **STOP POINTS** below, or a missing user-only value you cannot invent (HF username, GitHub URL, author details) when it is first needed.
- Do not rely on memory for library APIs (especially `transformers` cache classes such as `DynamicCache`, `peft`, `gradio`, which change between versions). Read the installed source/docstrings (`uv run python -c "import X; help(X.Y)"` or open the file in `.venv`) and prove behaviour with the tiny-model tests.

#### A.2.1 Self-review at the end of every phase (replaces user review)
Write a short checklist result in the `PROGRESS.md` session log: (1) quality gate output green; (2) R1-R10 each re-checked on the files touched this phase (run the greps/commands in section 0); (3) every phase checklist item ticked only with evidence (test name or file path); (4) no file outside the layout; (5) no number or claim that is not backed by a file in `results/` or a passing test. If any check fails, fix it before moving on.

#### A.2.2 STOP POINTS (the only planned pauses)
- **STOP POINT 1: end of Phase 3 (training code ready).** Finish Phase 3, write the Colab commands for jobs 1-3 (train, val evaluation) into README + `PROGRESS.md -> Awaiting user`, commit, then **stop** with a short message telling the user exactly what to run on Colab (copy-paste commands, expected runtime, what to bring back/push). The user starts training and tells you to continue; training runs in the background while you code Phases 4-7, which do not depend on the trained adapter (use tiny random models on CPU for tests, the base model for smoke checks).
- **STOP POINT 2: end of Phase 7 (all code done).** Before stopping, also complete everything in Phase 9 that needs no results (LaTeX skeleton with every section, `refs.bib`, README, `docs/`). Then stop with the consolidated list of remaining jobs (jobs 3-6: finish evaluation incl. the single `--final` test run, benchmarks, adapter upload) and user-only tasks. After the user returns results, record them in `PROGRESS.md`, do the error analysis, fill the report numbers, fix any bugs revealed by real runs, update the final R1-R10 audit, and finish.
- Between stop points, if you finish early or hit a result you need from the user, update `PROGRESS.md` and stop; otherwise keep going.

### A.3 `PROGRESS.md` format (create in Phase 1, keep it short and factual)

```markdown
# PROGRESS

Last updated: <ISO timestamp> by <agent/model name>

## NEXT ACTION
<one or two precise sentences: exact file/function/command to do next>

## Phase status
| Phase | Status (todo / in-progress / done / awaiting-user) | Commit/tag | Notes |
|-------|-----|-----|-----|
| 1 Setup | ... | ... | ... |
... one row per phase (1-9)

## Checklist state
(Copy each phase's checkboxes here; tick as done. Keep wording identical to PLAN.md.)

## Awaiting user (jobs only the user can run / things needed)
- [ ] e.g. "Run Colab training: command in README, section Long-running jobs, job 1"
- Results received from user: <file paths in results/, or "none yet">

## Facts discovered (dataset columns, label strings, token stats, versions, measured timings)
...

## Open problems / blockers
...

## Session log (append-only, newest last)
- <timestamp> <agent>: <what was done, what was left unfinished, any surprises>
```

### A.4 Rules for `PROGRESS.md`
- It is the single source of truth for status. `PLAN.md` is never edited by agents except to fix a clear error (log that in the session log).
- A checkbox is ticked only when the item is done **and verified** (test passed / command ran / file exists). Never tick on intent.
- "Awaiting user" items (Colab runs, adapter upload, screenshots, video) must be written there the moment the code for them is ready, with the exact command. When the user returns results, record the paths and what was learned.
- Never put secrets (HF token) anywhere in the repo or in `PROGRESS.md`.

### A.5 User prompts (for reference)
- **First ever session:** "Follow AGENTS.md and PLAN.md section A. HF adapter repo: `<HF_USERNAME>/trialmatch-qwen-lora`; GitHub repo: `<URL>`; author: `<name, affiliation, email>`. Work autonomously from Phase 1 and stop only at STOP POINT 1."
- **After starting Colab training:** "Training is running. Continue from NEXT ACTION through Phase 7 and stop at STOP POINT 2."
- **Switching agent/model or after a token cutoff:** "Follow AGENTS.md: read PROGRESS.md and continue from NEXT ACTION."
- **After Colab results are back:** "Results are in `results/` (pushed). Finish the remaining work per STOP POINT 2."

---

## 0. Non-negotiable rules (from the user). Verify before finishing every phase

| # | Rule | How it is enforced / checked |
|---|------|------------------------------|
| R1 | **Use `uv`** for everything (env, deps, running). Never `pip install` or bare `python`. | `uv init`, `uv add`, `uv run ...`. Commit `pyproject.toml` + `uv.lock`. README shows only `uv` commands. |
| R2 | **Always use type annotations** (params, returns, attributes, locals when non-obvious). | `uv run mypy --strict src tests` has 0 errors. |
| R3 | **Minimize global variables.** No mutable module-level state; no `global` keyword. | `grep -rn "\bglobal " src` returns nothing. Dependencies are passed via constructors. |
| R4 | **All constant values live in `src/trialmatch/constants.py`** and are imported from there (model names, paths, seeds, label sets, hyperparameter defaults, prompt templates, tolerances, benchmark grids, adapter repo name). No magic numbers or strings elsewhere. | Review with grep for literals like `42`, `"Qwen`, `"results/`. |
| R5 | **Don't optimize prematurely.** Plain, correct implementations first. No custom CUDA, no quantization, no batching tricks unless a measured problem demands it. | Justify any optimization in a comment citing a measurement. |
| R6 | **Minimal code.** If 50 lines do the job, don't write 1000. Prefer library calls (datasets, peft, transformers `Trainer`, sklearn) *except* for parts the assignment says must be hand-implemented (KV decoding loop, tiled attention, speculative loop). No speculative abstractions. | `src/` should be roughly under 1,200 lines. Delete dead code. |
| R7 | **Write documentation.** Google-style docstring on every public class/function, a module docstring per file, `README.md`, `docs/`. | `uv run ruff check --select D` passes. |
| R8 | **Do NOT run long jobs yourself** (training, full evaluation, full benchmarks, anything over about 10 minutes). Write the script, run only a smoke test (`--smoke`, under 2 min), and document the exact command, expected runtime and outputs in `README.md` under "Long-running jobs (run on Colab)". Also add each to `PROGRESS.md -> Awaiting user`. | See section 10. |
| R9 | **Consistent coding style** across all files. | `ruff format` + `ruff check` (config in `pyproject.toml`) clean before every commit. Google docstrings, double quotes, 100-col lines, `from __future__ import annotations` in every module. |
| R10 | **Use classes.** Logic is organized as small classes (stateless helpers are classes with `@staticmethod`s). Config is frozen `@dataclass`es. Entry points are thin `main()` functions that instantiate classes. | No loose functions except `main()` and test functions. |

R6 and R10 pull against each other: resolve with **small, thin classes**, never class hierarchies unless the assignment needs polymorphism (the `Decoder` family does).

---

## 1. Decisions already made with the user

- **Compute**: **Google Colab free T4** (16 GB VRAM, fp16 only, no bf16, no FlashAttention-2). Colab sessions are short and can disconnect, and the disk is wiped at session end, so: **every long job must checkpoint and be resumable (`--resume`), and its outputs must be pushed/saved immediately** (Drive mount or `git push`/HF upload; see section 10).
- **Target model**: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, 1.54B params, 28 layers, GQA 12Q/2KV heads, 32k context, byte-level BPE vocab ~151k).
- **Draft model**: `Qwen/Qwen2.5-0.5B-Instruct` (same tokenizer, so speculative decoding is valid). A test verifies tokenizer equality.
- **Fine-tuning**: LoRA via `peft` on fp16 base weights (about 3 GB; no 4-bit, so no bitsandbytes).
- **UI**: Gradio.
- **Public assets (no manual download needed)**: the dataset and both Qwen models are public and ungated; the libraries fetch them automatically. If the agent's sandbox blocks internet, the user pre-downloads them with `hf download` and the agent reuses the HF cache.
- **User-supplied values** (placeholders the agent must put in `constants.ADAPTER_REPO` and the README; ask the user if still unknown): HF adapter repo `<HF_USERNAME>/trialmatch-qwen-lora`; GitHub repo URL; author name/affiliation/email for the paper. The HF token is **never** given to the agent; the user uses it only in Colab secrets / local `hf auth login`.
- **Report build**: the user compiles `report/` on **Overleaf** (upload `main.tex`, `refs.bib`, `figures/`). The agent only writes valid IEEEtran LaTeX and does not need a local TeX install.
- **Rubric note**: rubric rows sum to 135 though the header says 100. Treat every row as required.
- **Explanations**: the dataset has no human explanations (only GPT-4 ones, forbidden as input). The model outputs `label` + `evidence_ids` only. The **explanation is generated deterministically** by a template from label + extracted evidence text. When evidence is missing, the template must say "no supporting evidence cited; this is not proof of eligibility or ineligibility".

Second candidate for A2 (comparison only, not trained): `meta-llama/Llama-3.2-1B-Instruct` or `Qwen/Qwen2.5-3B-Instruct`. Report params, license, fp16 memory (params x 2 bytes + KV cache), context length, and why Qwen2.5-1.5B was chosen.

---

## 2. Repository layout (create exactly this; no extra files)

```
trialmatch/
├── pyproject.toml            # uv project, ruff/mypy/pytest config
├── uv.lock
├── AGENTS.md                 # provided by the user; auto-loaded by opencode; short rules + handoff protocol
├── PLAN.md                   # this file
├── PROGRESS.md               # live status / handoff log (section A)
├── README.md                 # setup, long jobs (Colab), app startup, results table, AI-use pointer
├── docs/
│   ├── ARCHITECTURE.md       # module/class map, data flow
│   ├── DECISIONS.md          # alternatives considered/rejected, difficulties, R1-R10 audit (rubric A3)
│   └── AI_USE.md             # AI-use appendix
├── splits/split_manifest.json
├── results/                  # JSON/CSV/logs from Colab runs (committed, small)
├── src/trialmatch/
│   ├── __init__.py
│   ├── constants.py          # ALL constants (R4)
│   ├── data.py               # AnnotationLoader, SplitManager, PromptBuilder, SentenceSplitter
│   ├── schema.py             # Prediction dataclass, OutputParser
│   ├── modeling.py           # ModelLoader (base/adapter/merged, device/dtype)
│   ├── train.py              # FineTuner + main()
│   ├── evaluate.py           # Evaluator + main()
│   ├── decoding.py           # Decoder ABC, UncachedDecoder, CachedDecoder
│   ├── attention.py          # ReferenceAttention, TiledAttention
│   ├── speculative.py        # SpeculativeDecoder
│   ├── benchmark.py          # Benchmarker + main()
│   ├── pipeline.py           # ScreeningPipeline, OverallAggregator, ReportExporter
│   └── app.py                # ScreeningApp (Gradio) + main()
├── tests/                    # pytest; small, fast, CPU-only with tiny shapes
│   ├── test_data.py
│   ├── test_schema.py
│   ├── test_attention.py
│   ├── test_decoding.py
│   └── test_speculative.py
└── report/
    ├── main.tex              # IEEEtran conference
    ├── refs.bib
    └── figures/              # generated by evaluate/benchmark mains
```

(`PROGRESS.md` and `AGENTS.md` are the only additions for handoff.) Add `[project.scripts]` only if it removes code; otherwise `uv run python -m trialmatch.<module>`.

---

## 3. Phase 1: Project setup (about 0.5 h)

### 3.1 Tasks
- [ ] `uv init --package trialmatch`; Python 3.11.
- [ ] `uv add torch transformers peft datasets accelerate numpy scikit-learn gradio pandas matplotlib huggingface_hub`; `uv add --dev pytest ruff mypy`.
- [ ] Configure in `pyproject.toml`: ruff (format + lint incl. `D`, `I`, `ANN`, `UP`), mypy `strict = true`, pytest.
- [ ] `constants.py` skeleton with sections: paths, seeds, models, labels, training, prompt, decoding, benchmark, tolerances, app, `ADAPTER_REPO`. Use `Final` constants or frozen dataclass instances (no mutation).
- [ ] `AGENTS.md` is provided by the user in the repo root (opencode auto-loads it); do not overwrite it. Create `PROGRESS.md` (per A.3, pre-filled with all phases, checklists, and both STOP POINTS), `docs/DECISIONS.md`, a README skeleton, `.gitignore` (venv, caches, `*.safetensors`, `checkpoints/`, `__pycache__`, `.env`).
- [ ] Pin the versions resolved in `uv.lock`; record the `torch`/`transformers`/`peft`/`gradio` versions in `PROGRESS.md -> Facts discovered`.
- [ ] First commit + tag `phase-1-done`; add the GitHub remote if the user gave a URL (`git remote add origin ...`); do not push without being asked, unless credentials already work.

### 3.2 Quality gate (run after every phase; copy into README)
```
uv run ruff format . && uv run ruff check . && uv run mypy --strict src tests && uv run pytest -q
```

## 4. Phase 2: Data (rubric B1, B2). About 3 h

First **inspect the real dataset** (`ncbi/TrialGPT-Criterion-Annotations`) and record the true column names, label strings and note format in `PROGRESS.md -> Facts discovered` and `docs/DECISIONS.md`. Expected (verify, don't assume): columns for patient id, note, trial id, criterion type, criterion number/text, `gpt4_*` fields, `expert_sentence_ids`, `expert_eligibility`; 1,015 rows over 53 patients; inclusion labels {included, not included, not enough information, not applicable}; exclusion labels {excluded, not excluded, not enough information, not applicable}. Notes are sentence-numbered. If reality differs, adapt simply and tell the user.

- [ ] `AnnotationLoader.load() -> list[Annotation]`: load with `datasets`, **drop every `gpt4_*` and `explanation_correctness` field at load time** (leakage guard; test asserts they never appear in `Annotation` or any prompt).
- [ ] `SplitManager`: sorted unique patient IDs -> `np.random.default_rng(42).shuffle` -> 37/8/8. Save `splits/split_manifest.json` (IDs per split, counts, seed, row counts). `assert_no_leakage()` checks disjoint and complete. Test in `tests/test_data.py`.
- [ ] `SentenceSplitter`: for the dataset use the existing numbering; for new user notes in the app, split with a simple regex and number from 0, using the **same numbering format** the model is trained on.
- [ ] `PromptBuilder`: prompt = system instruction + numbered patient sentences + criterion type + criterion text + output-schema instruction. Target = compact JSON `{"label": "...", "evidence_ids": [3, 7]}` (empty list when none). Templates in `constants.py`.
- [ ] **Truncation analysis**: tokenize all prompts, report max/mean/p95 token length per split, set `MAX_SEQ_LEN` (start at 2048). Count examples exceeding it and whether gold evidence sentences fall in any truncated region; never silently drop gold-evidence sentences. Output `results/truncation_report.json` (for the paper).
- [ ] `uv run python -m trialmatch.data --summary` prints split counts, label distribution per split and criterion type, token stats. (Short; run it.)
- [ ] Self-review (A.2.1), update `PROGRESS.md`, commit, tag `phase-2-done`, continue to Phase 3.

## 5. Phase 3: Fine-tuning and evaluation (rubric B3, B4, C1-C4). About 5 h coding; GPU runs by the user

- [ ] `ModelLoader`: base fp16 (`device_map` auto), optional LoRA adapter, optional `merge_and_unload()` for decoding work. Uses CPU/fp32 when no GPU.
- [ ] `FineTuner` (HF `Trainer` + `peft.LoraConfig`; no hand-written loop). Config in `constants.py`: LoRA r=16, alpha=32, dropout=0.05, targets q,k,v,o,gate,up,down; lr 2e-4 cosine, warmup 3%, 3 epochs, effective batch 16 (grad accumulation), fp16, gradient checkpointing, **loss on completion tokens only** (prompt tokens = -100), eval each epoch on validation, `save_strategy="epoch"`, `save_total_limit=2`, `load_best_model_at_end` on validation loss. Seeds from `constants`.
- [ ] **Colab-safe**: `--output-dir` (default a Drive path when mounted), **`--resume`** flag (`resume_from_checkpoint=True`) so a dropped session continues.
- [ ] Log `trainer.state.log_history` to `results/train_log.json`; plot train/val loss curves to `report/figures/train_curves.pdf`.
- [ ] `--smoke`: 2 steps, 8 examples, temp output dir. This is all you run yourself.
- [ ] Optional `--sweep` (at most 4 configs, e.g. LoRA r in {8,16} x lr in {1e-4,2e-4}), chosen by **validation macro-F1 only**.
- [ ] `OutputParser`: strict JSON parse with tolerant fallback. Invalid output counts as a wrong label (`"invalid"`); report parse-failure rate. Evidence IDs outside the patient's sentence range are dropped and counted.
- [ ] `Evaluator` (sklearn):
  - Classification: accuracy, macro-F1, per-class P/R/F1, confusion matrix, each **overall, inclusion-only, exclusion-only**. Preserve all 4 label categories per type; **never collapse to binary**.
  - Evidence: per-record P/R/F1 on sentence-ID sets; **micro** (pooled TP/FP/FN) and **macro** (mean of per-record). Empty-set rule (document in README and paper): **gold empty and pred empty -> P=R=F1=1; exactly one side empty -> P=R=F1=0**. Also report metrics restricted to records with non-empty gold evidence.
  - Write `results/{split}_{model_tag}_{decode}.json`, predictions JSONL, confusion-matrix figures to `report/figures/`.
- [ ] Before/after: **base model (same prompt, greedy, same max_new_tokens, same parser)** vs **fine-tuned** on validation, then on test. Test is evaluated **once**, at the end, via `--split test --final`; the code must refuse `--split test` without `--final`. Never use test for choosing prompts, thresholds, hyperparameters or checkpoints.
- [ ] Error-analysis helper: dump >=12 misclassified test examples (stratified: wrong label, hallucinated evidence, missing evidence, not-applicable vs not-enough-info confusion, contradictory evidence) to `results/error_cases.md` with the cited patient sentences. The paper uses >=8 distinct ones.
- [ ] `OverallAggregator` (in `pipeline.py`; documented as an explicit rule, **not** a ground-truth label): any inclusion=`not included` or exclusion=`excluded` -> "Likely ineligible"; else any `not enough information` -> "Needs more information"; else all inclusion=`included` and exclusions in {`not excluded`,`not applicable`} -> "Likely eligible". `not applicable` criteria are ignored.
- [ ] Write the Colab jobs 1-3 into README + `PROGRESS.md -> Awaiting user` so the user can start training immediately while later phases are coded.
- [ ] Self-review (A.2.1), update `PROGRESS.md`, commit, tag `phase-3-done`, then **STOP POINT 1** (A.2.2).

## 6. Phase 4: KV-cached decoding (rubric D1-D3). About 4 h

`decoding.py`:
- [ ] `Decoder` ABC: `generate(input_ids, max_new_tokens) -> GenerationResult` (dataclass: `tokens`, `prefill_s`, `decode_s`, `peak_mem_bytes`, `n_forward_calls`).
- [ ] `UncachedDecoder`: each step runs `model(input_ids=full_sequence, use_cache=False)`, `argmax` of the last logit. Reference path.
- [ ] `CachedDecoder`: **explicit loop**: (1) prefill: one forward over the prompt with `use_cache=True`, keep `past_key_values`; (2) each step feed only the last token with `past_key_values`, explicit `position_ids = [[current_len]]`, `attention_mask` of length `cache_len + 1`; (3) update the returned cache; (4) stop on EOS or max tokens. **Do not call `model.generate`.** Comments explain positional handling (RoPE uses absolute index), causal masking (one query attends to all cached keys), cache growth (layers x 2 x kv_heads x head_dim x seq per sequence; GQA makes it 6x smaller here), and numerical-difference sources (fp16 reduction order, different matmul shapes).
- [ ] Timing: `torch.cuda.synchronize()` around timed regions, `torch.cuda.reset_peak_memory_stats()`, 2 warm-ups then >=5 repeats, report mean +/- std; CPU fallback uses `perf_counter`.
- [ ] Tests (`test_decoding.py`): tiny random Qwen2 config on CPU fp32; assert greedy equality cached vs uncached for **3 prompt lengths x 2 output lengths**. On Colab with the real fp16 model, record mismatches and investigate (first-diverging-step logit diff; show fp32 matches if it is a near-tie).
- [ ] `Benchmarker` decoding grid: input lengths {128, 512, 1024} x output lengths {32, 64, 128} (in `constants.py`) -> `results/bench_decoding.json` + figure.
- [ ] Self-review (A.2.1), update `PROGRESS.md`, commit, tag `phase-4-done`, continue to Phase 5.

## 7. Phase 5: Attention (rubric E1-E3). About 3 h

`attention.py` (pure PyTorch, shapes `(B, H, T, D)`):
- [ ] `ReferenceAttention`: `softmax(QK^T / sqrt(D) + causal_mask)V` with stable softmax (state whether manual max-subtraction or `torch.softmax`).
- [ ] `TiledAttention(block_q, block_k)`: per query block, loop key blocks keeping running max `m`, normalizer `l`, accumulator `acc`: `m_new = max(m, rowmax(S))`; `acc = acc*exp(m-m_new) + exp(S-m_new)@V`; `l = l*exp(m-m_new) + rowsum(exp(S-m_new))`; output `acc/l`. Skip fully masked blocks when causal. Handle `-inf` rows without NaN.
- [ ] Tests: shapes incl. non-multiples of the block size (e.g. T=37, block 16), causal and non-causal, fp32 `atol=1e-5`, fp16 `atol=1e-2`; compare against Reference **and** `torch.nn.functional.scaled_dot_product_attention`.
- [ ] Benchmark: lengths {256, 512, 1024, 2048}: max/mean abs error, runtime, peak memory for Reference, Tiled, PyTorch SDPA (try math/efficient/flash backends via `torch.nn.attention.sdpa_kernel`; report any unsupported on T4). Explain honestly why the Python-loop tiling is slower than a fused kernel (launch overhead, no SRAM fusion) and where it saves memory (no T x T matrix). Output `results/bench_attention.json` + figure.
- [ ] Self-review (A.2.1), update `PROGRESS.md`, commit, tag `phase-5-done`, continue to Phase 6.

## 8. Phase 6: Speculative decoding (rubric F1-F3). About 4 h

`speculative.py`: `SpeculativeDecoder(target, draft, gamma)` subclasses `Decoder`.
- [ ] Assert tokenizers identical (vocab and special tokens) in `__init__`; embedding sizes may differ (padded vocab), fine if ids map identically.
- [ ] Greedy algorithm: prefill both models; loop: draft proposes `gamma` tokens with its own cache; target runs **one** forward over `[last_accepted] + draft_tokens` using its cache; compare target argmax at each position with the draft token; accept the longest matching prefix; at the first mismatch (or after all accepted) append the **target's own token**; crop both caches to the accepted length (`DynamicCache.crop`) and re-sync the draft cache with any token it hasn't seen. Handle EOS inside accepted tokens and `max_new_tokens`.
- [ ] Record: acceptance rate (accepted/drafted), mean accepted length per round, rounds, target forward count, latency, tokens/s, peak memory (both models resident).
- [ ] Tests: tiny random models (CPU fp32) assert **exact token equality with greedy target decoding** for gamma in {1,3,5} and several prompts, plus cache lengths consistent after every round. Real-model check on Colab, plus a reference comparison with `model.generate(assistant_model=draft)`.
- [ ] Benchmark with real eligibility prompts from validation, gamma in {2,4,6}; report speedup **or slowdown** with explanation. Optional: LoRA-fine-tune the draft with the same `train.py` (`--model draft`) and show the acceptance-rate change. Output `results/bench_speculative.json`, figure.
- [ ] Self-review (A.2.1), update `PROGRESS.md`, commit, tag `phase-6-done`, continue to Phase 7.

## 9. Phase 7: Application (rubric G1-G4). About 5 h

`pipeline.py`:
- [ ] `ScreeningPipeline(model, tokenizer, decoder, splitter, builder, parser)`: `screen(note, criteria) -> ScreeningResult`, per criterion: criterion, label, evidence ids, evidence text (looked up from the **original** numbered sentences), template explanation, decoding mode and latency. Plus the overall outcome from `OverallAggregator`.
- [ ] `ReportExporter`: Markdown (and HTML) report: disclaimer, timestamp, model + adapter + decoding mode, patient sentences, per-criterion table, evidence quotes, overall preliminary outcome with the aggregation rule stated, missing-information list, provenance block.

`app.py` (`ScreeningApp`, Gradio Blocks, `uv run python -m trialmatch.app`):
- [ ] Persistent **disclaimer banner**: "Educational screening prototype, not a clinical enrollment decision."
- [ ] **Screening tab**: patient note (textbox or `.txt` upload); criteria as two textboxes (inclusion, exclusion; one per line) **or** a dropdown of example trials built from the dataset's test-split criteria (no external API needed); decoding mode dropdown (Uncached / KV-cache / Speculative); Run; results table (criterion, type, label, evidence IDs, evidence text, explanation); overall outcome; missing-information list; **Download report** button.
- [ ] **Benchmark tab**: live "Run comparison on this input" (runs all 3 decoders on the current prompt; table + charts for latency, tokens/s, peak memory, acceptance rate, token-equality check) plus a viewer for the stored Colab benchmark JSONs. Not static screenshots.
- [ ] Input validation and errors: empty note, no criteria, prompt over `MAX_SEQ_LEN`, unparseable model output (show raw output, mark invalid, continue), model not loaded. Friendly messages, never stack traces; the app stays usable afterward.
- [ ] Model loading: download the adapter from the HF Hub (`constants.ADAPTER_REPO`) if not at `constants.ADAPTER_DIR`; auto-select GPU or CPU (fp32 on CPU, about 6 GB RAM). Lazy-load once into a `ScreeningApp` instance attribute (no globals).
- [ ] README "Run the app": `uv sync`, `uv run python -m trialmatch.app`, URL, first-load time, CPU/GPU notes, how to use a local adapter. The evaluator must not need Colab or a notebook to use it.
- [ ] Smoke-test the UI wiring with `--smoke` (base 0.5B model, no adapter).
- [ ] Self-review (A.2.1), update `PROGRESS.md`, commit, tag `phase-7-done`. Then do the results-independent parts of Phase 9 and Phase 8 docs (complete the Colab job list in README), and **STOP POINT 2** (A.2.2).

## 10. Phase 8: Long-running jobs for the user (R8)

You **do not run** these. Write them into `README.md` under **"Long-running jobs (run on Colab T4)"**, each with command, approximate runtime, output files and where to save/commit them, and mirror them in `PROGRESS.md -> Awaiting user`.

**Colab setup cell block (document in README):** runtime = T4 GPU; add `HF_TOKEN` in the Colab secrets (key icon) and enable notebook access; `!pip install uv` (the only allowed pip use, to bootstrap uv); `!git clone <repo>`; `%cd` into it; `!uv sync`; mount Drive (`from google.colab import drive`) and point `--output-dir` at a Drive folder. After each job: `git add results report/figures && git commit && git push` (use a GitHub token via Colab secrets) **or** copy outputs to Drive immediately.

1. `uv run python -m trialmatch.train` (about 30-60 min est.; give your estimate from the smoke run). Resume after a disconnect with `--resume`.
2. `uv run python -m trialmatch.train --sweep` (optional)
3. `uv run python -m trialmatch.evaluate --split val --model base` and `--model finetuned`
4. `uv run python -m trialmatch.evaluate --split test --model base --final` and `--model finetuned --final` (**once**)
5. `uv run python -m trialmatch.benchmark --suite decoding|attention|speculative`
6. Upload the adapter to the HF Hub: `uv run hf upload <HF_USERNAME>/trialmatch-qwen-lora <adapter_dir>` (token from Colab secrets), and make sure `constants.ADAPTER_REPO` matches.

## 11. Phase 9: Report, docs, repo hygiene (rubric H1-H3). About 5 h

- [ ] `report/main.tex` in **IEEEtran** (conference): Title/authors, Abstract, Keywords, Introduction, Related Work, Materials and Methods (dataset, split, model, fine-tuning, architecture diagrams, attention/decoding algorithms as `algorithm` blocks, app workflow), Experimental Setup and Metrics, Results and Discussion, Limitations, Conclusion, References, AI-use appendix. Every figure/table is `\ref`'d **and interpreted** in the text. Include training curves, confusion matrices, before/after table, the >=8-error analysis, decode/attention/speculative benchmarks, and screenshot placeholders for the user. Must compile on Overleaf with only standard packages.
- [ ] `refs.bib`: >=5 **verified** real sources (verify each via web search; never invent citations). Suggested: TrialGPT (Jin et al., Nature Communications 2024); LoRA (Hu et al., 2022); FlashAttention (Dao et al., 2022); online softmax (Milakov & Gimelshein, 2018); speculative decoding (Leviathan et al., 2023; Chen et al., 2023); Attention Is All You Need (Vaswani et al., 2017); Qwen2.5 technical report; a zero-shot LLM trial-matching paper (e.g. Wornow et al., 2024).
- [ ] Numbers in the paper come **only from `results/*.json`**; never invent numbers. Leave `TODO(user)` markers where results don't exist yet.
- [ ] `docs/DECISIONS.md`: alternatives considered/rejected, difficulties (rubric A3), R1-R10 audit table (one line per rule with evidence). `docs/AI_USE.md`: honest AI-assistance account (list the agents/models used, per `PROGRESS.md` session log).
- [ ] README: overview, repo map, setup, reproduction (all long jobs), results summary table, split manifest location, model download link, app startup, demo video link (placeholder), disclaimer.
- [ ] Final gate: ruff/mypy/pytest green; `git status` clean; no large weights committed; `PROGRESS.md` shows all phases done and "Awaiting user" lists only the user-only items below.

---

## 12. Suggested schedule (today is Nov 4; deadline Nov 10)

| Day | Work |
|-----|------|
| Nov 4 | Phases 1-2 |
| Nov 5 | Phase 3 code + smoke; **user starts Colab training** |
| Nov 6 | Phases 4-5; user runs val/test evaluation and cache/attention benchmarks |
| Nov 7 | Phase 6 + speculative benchmarks |
| Nov 8 | Phase 7 app; user uploads the adapter, records the demo |
| Nov 9 | Phase 9 report fill-in (Overleaf), error analysis, repo cleanup |
| Nov 10 | Buffer; submit well before the deadline (the grace period is not a plan) |

## 13. Things only the user can do (list in `PROGRESS.md -> Awaiting user`)

Run the Colab jobs; upload the adapter to the HF Hub; push to GitHub; compile the report on Overleaf; take app screenshots; record the 5-7 min video; read and understand every component (the defense may require modifying cache/attention/speculative code live and testing unseen cases); finalize the AI-use appendix.

## 14. Agent working agreement

1. Follow section A on every session; work autonomously phase by phase; pause only at the two STOP POINTS (or for a missing user-only value).
2. Don't create files outside the layout in section 2.
3. If the dataset contradicts assumptions here, record it in `PROGRESS.md` and `docs/DECISIONS.md`, choose the simplest consistent option, and tell the user.
4. Never touch the test split except through `--final`. Never feed `gpt4_*` fields anywhere.
5. If a requirement can't be met on free hardware, document the limitation honestly rather than faking results.
6. Never put secrets in the repo. Never claim a result you haven't seen in a file under `results/`.
