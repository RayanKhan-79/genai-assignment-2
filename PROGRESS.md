# PROGRESS

Last updated: 2026-11-09T11:55 by opencode / DeepSeek V4.1 Flash

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
| 9 Report + docs + hygiene | nearly done | 79db646 | all tables/figures filled from `results/`; only app screenshots + video link (user) remain |

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
- [x] Final gate green, clean tree, no weights committed — 77 tests, `mypy --strict` clean, no `.safetensors`/`.bin` tracked (checked 2026-11-08).
- [x] Fill report numbers from `results/*.json`, interpret every figure/table, finish error analysis (jobs 3-5 ingested).
- [ ] Last `TODO(user)` in `report/main.tex` (two app screenshots) replaced after the user supplies them; then tag `submission`.

## Awaiting user (jobs only the user can run / things needed)
**Everything the agent can do is done (commit `79db646`). Three user-only items remain:**

1. [x] **HF adapter repository public.** Verified 2026-11-09: the model API returns 200 and
   `checkpoints/adapter` was downloaded locally (adapter_config.json + 74 MB adapter_model.safetensors).
   Job 6 is complete; the app now loads the fine-tuned adapter.
2. **Two app screenshots** -> `report/figures/app_screening.png` (screening tab with results) and
   `app_benchmark.png` (benchmark tab). Run `uv run python -m trialmatch.app` locally.
3. **Overleaf compile + video.** Compile `report/main.tex` (the last red `TODO(user)` is only the
   screenshot note; `\pendingfigure` already draws placeholders so it compiles without them), add the
   5-7 min video link to the README, and finalize `docs/AI_USE.md` if desired. Then ping the agent for
   the R1-R10 audit and the `submission` tag.

### Job status
- [x] Job 1 - training done on Colab (3 epochs, eval loss 0.1349/0.1170/0.1227, train loss 0.1022, 697 s). Session died before the log was pushed; `results/train_log.json` reconstructed from the preserved console output (D22), `report/figures/train_curves.pdf` regenerated.
- [x] Job 2 (optional sweep) - **skipped on purpose**; re-running would replace the adapter that jobs 3-5 evaluated.
- [x] Job 3 - validation eval: `val_{base,finetuned}_hf.json`, `val_finetuned_cached.json` (cached == hf byte-identical on 206/206).
- [x] Job 4 - test eval once (`--final`): `test_{base,finetuned}_hf.json`, `results/error_cases.md` (92 cases).
- [x] Job 5 - all three benchmark suites: `bench_{decoding,attention,speculative}.json`; figures moved to `report/figures/` (D21).
- [x] Job 6 - adapter uploaded and the repository made public; verified 2026-11-09 (`/api/models` -> 200) and downloaded into `checkpoints/adapter`.
