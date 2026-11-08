# PROGRESS

Last updated: 2026-11-08T19:20 by opencode / DeepSeek V4.1 Flash

## NEXT ACTION
All Colab jobs (1, 3, 4, 5, 6) are done and every result table/figure in `report/main.tex` is now
filled from `results/`. Remaining: the user makes the HF adapter repo public, adds the two app
screenshots and the video link, compiles on Overleaf; then the agent runs the final R1-R10 audit,
cleans the tree and tags the submission.

## Results ingested (jobs 1, 3, 4)

| Run | Accuracy | Macro F1 | Evidence micro / macro F1 | Invalid |
|-----|----------|----------|---------------------------|---------|
| val base hf | 0.0146 | 0.0089 | 0.101 / 0.167 | 0.0 |
| val finetuned hf | 0.7039 | 0.3380 | 0.569 / 0.761 | 0.0 |
| val finetuned cached | 0.7039 | 0.3380 | 0.569 / 0.761 | 0.0 |
| test base hf | 0.0194 | 0.0529 | 0.050 / 0.184 | 0.0 |
| test finetuned hf | 0.7136 | 0.3730 | 0.543 / 0.803 | 0.0 |

- `val_finetuned_cached` raw outputs are byte-identical to `val_finetuned_hf` on all 206 records
  (hand-written KV cache == `model.generate` on the real fp16 model).
- Fine-tuned model never predicts `not included`, `not applicable` or `excluded`; base model predicts
  only `not applicable` and `excluded` (plus one `not included` on test).
- Job 5 (T4, fp16): decoding is token-identical on all 9 grid points, speedup 1.04-5.11x; attention
  tiled max error <=2.1e-3, 26 MiB vs 158 MiB reference at 2048 but 62x slower, FlashAttention
  unsupported on sm_75, SDPA efficient fastest; speculative is lossless (0 mismatches) with
  acceptance 0.82/0.70/0.63 but latency 7.04/7.34/7.58 s vs 4.81 s baseline (0.64-0.68x).
- Job-5 figures were written to `results/` by the old `_figure_root`; fixed to `report/figures/`
  and the three PDFs moved there.
- HF adapter repo `RayanKhan-79/genai-assignment-2` returns HTTP 401: it is private. It must be made
  public or the app download silently falls back to the base model.
- Error dump: 92 cases (wrong_label 56, missing_evidence 16, unsupported_evidence 9, extra_evidence
  8, not_applicable_vs_not_enough_information 3).

## Phase status
| Phase | Status | Commit/tag | Notes |
|-------|--------|------------|-------|
| 1 Setup | done | 2223d03 / `phase-1-done` | uv project, deps, ruff/mypy/pytest config, constants skeleton, docs |
| 2 Data | done | ebf0ba1 / `phase-2-done` | loader, splits, prompts, truncation report |
| 3 Fine-tuning + evaluation | awaiting-user | 9f3995a / `phase-3-done` | code + smoke green; Colab jobs 1-3 handed to user (STOP POINT 1) |
| 4 KV-cached decoding | done | f4cc341 / `phase-4-done` | Decoder ABC, Uncached/Cached loops, decoding benchmark |
| 5 Attention | done | 82af14b / `phase-5-done` | reference + tiled online softmax, 11 tests, benchmark suite |
| 6 Speculative decoding | done | f7f6a12 / `phase-6-done` | draft/verify loop with cache re-sync, 13 tests, benchmark suite |
| 7 Application | done | 8365cf6 / `phase-7-done` | pipeline + report exporter + Gradio app, smoke green |
| 8 Long-running jobs | done (docs) | ba558b7 | jobs 1-6 in README + Awaiting user; user runs them |
| 9 Report + docs + hygiene | nearly done | e979e08 | all tables/figures filled from `results/`; only app screenshots + video link (user) remain |

## Checklist state
### Phase 1
- [x] `uv init --package trialmatch`; Python 3.11.
- [x] `uv add torch transformers peft datasets accelerate numpy scikit-learn gradio pandas matplotlib huggingface_hub`; `uv add --dev pytest ruff mypy`.
- [x] Configure in `pyproject.toml`: ruff (format + lint incl. `D`, `I`, `ANN`, `UP`), mypy `strict = true`, pytest.
- [x] `constants.py` skeleton (paths, seeds, models, labels, training, prompt, decoding, benchmark, tolerances, app, `ADAPTER_REPO`).
- [x] `PROGRESS.md`, `docs/DECISIONS.md`, README skeleton, `.gitignore`.
- [x] Record dependency versions in `Facts discovered`.
- [x] First commit + tag `phase-1-done`; add GitHub remote.

### Phase 2
- [x] `AnnotationLoader.load()` drops every `gpt4_*` / `explanation_correctness` field.
- [x] `SplitManager` 37/8/8 by patient, `splits/split_manifest.json`, `assert_no_leakage()`.
- [x] `SentenceSplitter` (dataset numbering + regex fallback for app notes).
- [x] `PromptBuilder` (system + numbered sentences + criterion + schema; compact JSON target).
- [x] Truncation analysis -> `results/truncation_report.json`.
- [x] `uv run python -m trialmatch.data --summary` runs.
- [x] Self-review, commit, tag `phase-2-done`.

### Phase 3
- [x] `ModelLoader` (base/adapter/merged, device+dtype) — `modeling.py`, smoke-tested.
- [x] `FineTuner` (Trainer + peft LoRA, completion-only loss, `--resume`, `--output-dir`, `--smoke`) — `train.py --smoke` ran 2 steps on CPU (loss 0.608, adapter saved).
- [x] `results/train_log.json` + `report/figures/train_curves.pdf` — written by real runs; smoke writes into `checkpoints/smoke/`.
- [x] Optional `--sweep` (<=4 configs, validation macro-F1) — `Sweeper`, user-run only.
- [x] `OutputParser` (strict + tolerant fallback, invalid counts as wrong label) — `tests/test_schema.py` (9 tests).
- [x] `Evaluator` (classification overall/inclusion/exclusion, evidence micro/macro, empty-set rule) — `tests/test_evaluate.py`; `--smoke` run produced 4/4 strict JSON.
- [ ] Base vs fine-tuned on val (Colab job 3) — **awaiting user**.
- [ ] Test evaluated once via `--split test --final` (Colab job 4) — **awaiting user**.
- [x] Error-analysis helper -> `results/error_cases.md` (>=12 cases) — `ErrorAnalyzer`, `--errors`, user-run on test.
- [x] `OverallAggregator` documented rule — `pipeline.py`, `tests/test_pipeline.py`.
- [x] Colab jobs 1-3 in README + Awaiting user.
- [x] Self-review, commit, tag `phase-3-done` -> **STOP POINT 1**.

### Phase 4
- [x] `Decoder` ABC + `GenerationResult` — `decoding.py`.
- [x] `UncachedDecoder` reference path.
- [x] `CachedDecoder` explicit loop (no `model.generate`), documented masking/positions/cache math.
- [x] Timing with CUDA sync/peak memory, warm-ups + repeats — `Timing` class, `BenchmarkConfig`.
- [x] `tests/test_decoding.py` greedy equality 3 prompt lengths x 2 output lengths — 10 tests, fp32 CPU.
- [x] `Benchmarker` decoding grid -> `results/bench_decoding.json` + figure — smoke verified (tiny model, 1.2x speedup, tokens equal); real run = Colab job 5.
- [x] Self-review, commit, tag `phase-4-done`.

### Phase 5
- [x] `ReferenceAttention` — `attention.py`, `torch.softmax` (documented).
- [x] `TiledAttention` online softmax — running max/normaliser/accumulator, causal block skip, no NaN.
- [x] Tests: shapes incl. non-multiples (T=37), causal/non-causal, fp32/fp16, vs reference + SDPA — 11 tests.
- [x] Benchmark vs SDPA backends -> `results/bench_attention.json` + figure — smoke verified (error <= 1e-6, EFFICIENT_ATTENTION reported unsupported on MPS); real run = Colab job 5.
- [x] Self-review, commit, tag `phase-5-done`.

### Phase 6
- [x] `SpeculativeDecoder` (tokenizer equality assert, greedy accept/reject, cache cropping) — `speculative.py`, docstring states the cache invariant.
- [x] Acceptance-rate and latency metrics — `SpeculativeStats` (+ `GenerationResult.stats`).
- [x] Tests: exact equality with greedy target for gamma in {1,3,5} — `tests/test_speculative.py` (13 tests: lossless, full-acceptance cache alignment, EOS, tokenizer mismatch).
- [x] Benchmark -> `results/bench_speculative.json` + figure — smoke verified (tokens equal, HF `assistant_model` prefix check matches); real run = Colab job 5.
- [x] Self-review, commit, tag `phase-6-done`.

### Phase 7
- [x] `ScreeningPipeline.screen()` — `pipeline.py` (prompt per criterion, prompt-length guard, `assemble()` pure path).
- [x] `ReportExporter` (Markdown/HTML, disclaimer, provenance) — `pipeline.py`, 6 new tests in `tests/test_pipeline.py`.
- [x] Gradio app: disclaimer, screening tab, benchmark tab, error handling, lazy model load — `app.py` (`ScreeningApp`).
- [x] README "Run the app" — commands, first-load times, adapter resolution, tabs, error behaviour.
- [x] `--smoke` UI wiring test — `uv run python -m trialmatch.app --smoke` green (2 criteria, 1.20 s cached; 3 benchmark rows, 169-token prompt).
- [x] Self-review, commit, tag `phase-7-done` -> **STOP POINT 2**.

### Phase 8
- [x] Six Colab jobs documented in README + Awaiting user — jobs 1-6 with commands, runtimes, outputs and push instructions.

### Phase 9
- [x] `report/main.tex` (IEEEtran, all sections, algorithm blocks) — compiles with conditional figure placeholders.
- [x] `refs.bib` >=5 verified real sources — 8 entries, each checked against arXiv/publisher pages.
- [x] Numbers only from `results/*.json`; `TODO(user)` where missing — report contains no invented number.
- [x] `docs/DECISIONS.md`, `docs/AI_USE.md`, README complete — README has jobs 1-6, results table, app section.
- [ ] Final gate green, clean tree, no weights committed — gate green; final audit waits for real results.
- [ ] Fill report numbers from `results/*.json`, interpret every figure/table, finish error analysis (after jobs 3-6).

## Awaiting user (jobs only the user can run / things needed)
**STOP POINT 2 - remaining work, in order (exact commands in README "Long-running jobs"):**
- [x] Job 1 - training finished on Colab (3 epochs, train loss 0.1014, eval loss 0.1334/0.1194/0.1250, 708 s); adapter + `results/train_log.json` saved.
- [ ] Regenerate `report/figures/train_curves.pdf` after `git pull`: `uv run python -m trialmatch.train --plot-results results/train_log.json`. Then commit/push `results/` + `report/figures/`.
- [ ] Job 2 (optional) - sweep: `uv run python -m trialmatch.train --sweep`.
- [x] Job 3 - validation eval done: `results/val_{base,finetuned}_hf.json` and `val_finetuned_cached.json` (cached == hf byte-for-byte).
- [x] Job 4 - test eval done once: `results/test_{base,finetuned}_hf.json`, `results/error_cases.md` (206 + 206 records, 92 error cases).
- [ ] Job 5 - benchmarks: re-run all three after `git pull` (the speculative suite crashed on a
  transformers-5 `Cache.crop` sign bug, now fixed): `--suite decoding`, `--suite attention`,
  `--suite speculative --model finetuned --adapter ...`.
- [ ] Job 6 - upload adapter: `uv run hf upload RayanKhan-79/genai-assignment-2 /content/drive/MyDrive/trialmatch/adapter`.
- [ ] Push `results/` + `report/figures/` after each job (`git add results report/figures && git commit && git push`).
- [ ] Compile `report/main.tex` on Overleaf; replace every red `TODO(user)` placeholder; add screenshots (`report/figures/app_*.png`) and the 5-7 min video link.
- [ ] Finalize `docs/AI_USE.md` with anything the human wants to add.
- Results received from user: **none yet** (every result table is still a placeholder).

## Facts discovered
- Environment: macOS 26.6.2 arm64, uv 0.12.5, Python 3.11 (uv-managed).
- Resolved versions: torch 2.15.0, transformers 5.22.1, peft 0.21.2, gradio 6.31.0,
  datasets 5.1.0, huggingface_hub 2.4.0, numpy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1,
  matplotlib 3.11.2, ruff 0.17.3, mypy 2.4.1, pytest 9.2.0.
- `transformers` 5.x and `gradio` 6.x are newer than the versions assumed by PLAN.md: every API
  call must be verified against the installed source before use.
- MPS is available locally; CPU/MPS only, no CUDA. No local GPU.
- **Dataset** `ncbi/TrialGPT-Criterion-Annotations`: single split `train`, **1,015 rows**,
  **53 patients**, 103 trials. Columns: `annotation_id, patient_id, note, trial_id, trial_title,
  criterion_type, criterion_text, gpt4_explanation, explanation_correctness, gpt4_sentences,
  expert_sentences, gpt4_eligibility, expert_eligibility, training`. There is **no**
  `criterion_number` column (PLAN expected one) and no per-patient split column.
- `expert_sentences` / `gpt4_sentences` are JSON **strings** (`"[0, 1, 2]"`), not lists.
- `criterion_type` is lowercase `inclusion`/`exclusion`; labels are the expected 4 + 4.
  Distribution: included 150, not included 23, not enough information 174, not applicable 9;
  excluded 10, not excluded 484, not enough information 119, not applicable 46.
- Notes are pre-numbered `"0. sentence"` lines, always consecutive from 0 (verified on all 53).
- Split (seed 42, patient-level 37/8/8): **train 603 rows / val 206 / test 206**.
- Token stats (Qwen tokenizer, prompt+target): max 595, mean ~300, p95 474; **0 examples exceed
  MAX_SEQ_LEN=2048** -> `results/truncation_report.json`, no gold evidence truncated.
- Measured locally: `train --smoke` (0.5B, 2 steps, CPU) = 1 min 45 s wall; `evaluate --smoke`
  (0.5B, 4 val records, MPS) = 0.69 s/record, 100% strict JSON with the final prompt.
- Only 10 `excluded` and 23 `not included` examples exist, so those classes are rare; macro-F1
  over 4 classes per criterion type will be noisy on val/test (206 rows each).
- Speculative smoke (tiny 2-layer target, 1-layer draft, CPU): tokens equal to cached target,
  `assistant_model` prefix check matches, gamma=2 gives 1.0 acceptance (both random models emit
  the same constant token) — wiring check only, real numbers come from Colab job 5.
- App smoke (0.5B base, CPU/MPS): 2 criteria with KV cache in 1.20 s; live benchmark ran three
  decoders on a 169-token prompt with a 32-token budget; UI config has 37 components.
- Test suite at STOP POINT 2 + plotting fix: 75 tests, mypy --strict clean over 22 source files,
  ruff clean.
- `transformers` 5 `Cache.crop(k)` expects a **negative** `k` (positive raises); the speculative
  benchmark hit this on the first warm-up round because the previous tests never produced a rejected
  proposal (random tiny models collapse to one token, so every draft matched). Fixed; two new tests
  force total and partial rejection.
- Colab sets `MPLBACKEND=module://matplotlib_inline.backend_inline`; that module is not installed in
  the uv venv, so `import matplotlib` raised `ValueError: Key backend` after job 1 finished training.
  All plotters now set `MPLBACKEND=Agg` (constant `MPL_BACKEND`) before importing matplotlib.
- Job 1 real numbers (from the Colab log, files not yet pushed): train loss 0.1014, eval loss
  0.1334 (epoch 1), 0.1194 (epoch 2), 0.1250 (epoch 3), train runtime 708 s, 3 epochs.
- Explanations are deterministic: `APP_EXPLANATIONS[label]` + quoted cited sentences, or
  `APP_NO_EVIDENCE` (the wording PLAN section 1 requires) when nothing is cited.
- arXiv/publisher checks done this session for every entry in `report/refs.bib` (TrialGPT DOI
  10.1038/s41467-024-53081-z; Qwen2.5 2412.15115; LoRA 2106.09685; FlashAttention 2205.14135;
  online softmax 1805.02867; Leviathan 2211.17192; Chen 2302.01318; Vaswani 1706.03762).

## Open problems / blockers
- None yet.

## Session log (append-only, newest last)
- 2026-11-08T19:20 opencode: Ingested job 5 (decoding/attention/speculative) and filled every
  benchmark table and interpretation in the report from `results/bench_*.json`. Decoding: all nine
  grid points token-equal, 1.04-5.11x. Attention: tiled error <=2.1e-3, 26 vs 158 MiB at 2048 but
  62x slower than the reference; FlashAttention unsupported on sm_75; SDPA efficient fastest.
  Speculative: lossless (0 mismatches, HF assistant prefix matches) but 0.64-0.68x, an honest
  slowdown. Found and fixed a packaging bug: benchmark figures were written to `results/` instead of
  `report/figures/`; moved the three PDFs. Reconstructed the job-1 log from the console output the
  user supplied (second run: eval 0.1349/0.1170/0.1227, 697 s) and regenerated the curve; corrected
  the report's epoch numbers. The HF adapter repo is private (401) and must be made public.
- 2026-11-08T18:30 opencode: Ingested jobs 1, 3 and 4 from `origin/main` (commit `bd49b03`) and filled
  the validation/test classification, evidence and error tables in `report/main.tex` from
  `results/*.json` only. Fixed the job-5 crash: local tests had never exercised a rejected draft, so
  `Cache.crop(budget - matched)` passed a positive count, which transformers 5 rejects. The loop now
  passes zero/negative counts, the module docstring states the convention, and two new
  `ForcedDraft` tests force total and partial rejection (77 tests total). Speculative smoke green.
  Still pending: job 5 numbers/figures, job 6 adapter upload, screenshots/video, final audit.
- 2026-11-05T07:55 opencode: Fixed the job-1 crash. Training had completed and saved the adapter and
  `results/train_log.json`; only `TrainCurvePlotter.plot` failed because Colab exports the
  `matplotlib_inline` backend, absent from the uv venv. Every plotter (train, evaluation confusion
  matrices, three benchmark suites, app) now forces `MPLBACKEND=Agg` from `constants.MPL_BACKEND`
  before importing matplotlib, `_write_log` treats plotting as non-fatal, and
  `train --plot-results <train_log.json>` redraws the figure without retraining. New
  `tests/test_train.py` reproduces the Colab environment in a subprocess. Gate green: 75 tests,
  mypy clean over 22 files.
- 2026-11-05T03:20 opencode: STOP POINT 2 self-review (PLAN A.2.1). (1) Quality gate green: ruff
  format/check clean, `mypy --strict` clean over 21 files, **74 tests pass** (`uv run pytest`). (2)
  R1-R10 re-checked on the touched files: only `uv run` commands in README/scripts; every new
  function annotated (`pipeline.py`, `app.py`, `speculative.py`); no `global` and no mutable module
  state (`APP_EXPLANATIONS`/`APP_DECODE_LABELS` are `MappingProxyType`); every new constant
  (`APP_*`, `APP_NO_EVIDENCE`, `speculative_prompts`, `APP_BENCHMARK_MAX_NEW_TOKENS`) is in
  `constants.py`; no premature optimisation (plain PyTorch loops, justified tiling); no paper claim
  without a file in `results/` (all result cells are `\pending{}`); one ruff style; classes are
  thin frozen dataclasses; test split still guarded by `--final`; no `gpt4_*` field is read; no
  secret in the tree; `checkpoints/` and weights are ignored. (3) Every checked item has evidence in
  the file/test name listed in the checklists above. (4) No file outside the PLAN layout except the
  ignored `checkpoints/` smoke/report artefacts. (5) No unbacked number: every figure/table in
  `report/main.tex` is a red placeholder until the matching Colab job lands.
  Also committed the evidence-tied explanation change (`66a850b`) and updated this section for
  STOP POINT 2.
- 2026-11-05T02:55 opencode: Phase 8 docs + Phase 9 results-independent parts done. README now has
  jobs 1-6, the results table and a complete app section; added `docs/ARCHITECTURE.md`,
  `docs/AI_USE.md`, `report/main.tex` (IEEEtran skeleton, algorithms, all sections, conditional
  figure placeholders) and `report/refs.bib` (8 citations verified online). Gate green, 73 tests.
  Stopped at **STOP POINT 2**.
- 2026-11-05T02:10 opencode: Phase 7 done. `ScreeningPipeline` + `ReportExporter` (Markdown and
  HTML, disclaimer, rule, provenance), `app.py` with lazy merged-adapter target, speculative draft,
  screening tab, live three-decoder benchmark tab and stored-JSON viewer. App smoke green on the
  0.5B base model (2 criteria in 1.20 s, benchmark 3 rows). 73 tests green. Next: Phase 9
  results-independent parts, jobs 4/6 docs, STOP POINT 2.
- 2026-11-05T01:25 opencode: Phase 6 done. `speculative.py` (gamma budget, one target forward per
  round, cache crop by rejected proposals, one extra draft step on full acceptance, tokenizer
  equality check), 13 tests. Benchmark suite wired: 8 val prompts, gammas 2/4/6, fixed 128-token
  budget, acceptance/speedup/latency/memory, HF `assistant_model` cross-check. Smoke green
  (67 tests, tokens equal). Next: Phase 7.
- 2026-11-04T23:40 opencode: Phase 2-3 done. Phase 2: dataset inspected (facts above),
  `data.py` + `tests/test_data.py`, split manifest, truncation report. Phase 3: `schema.py`,
  `modeling.py`, `train.py`, `evaluate.py`, `pipeline.py` (aggregator) + tests. Smoke runs green:
  train 2 steps CPU (1:45 wall), eval 4 records 0.5B (strict JSON 4/4). Prompt reworded after the
  first template made the base model emit a bare `[]` (D7). STOPPED at STOP POINT 1.
- 2026-11-04T20:15 opencode: Phase 1 started. Created uv project (root moved from
  `trialmatch/` subdir to repo root), added runtime+dev deps, wrote `pyproject.toml`
  tooling config, `.gitignore`, `constants.py` skeleton. Next: README, DECISIONS, first commit.
