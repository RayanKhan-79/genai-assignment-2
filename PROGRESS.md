# PROGRESS

Last updated: 2026-11-04T20:15 by opencode / DeepSeek V4.1 Flash

## NEXT ACTION
Phase 1: finish project setup (README skeleton, docs/DECISIONS.md, first commit + tag
`phase-1-done`, add GitHub remote), then Phase 2 `src/trialmatch/data.py`.

## Phase status
| Phase | Status | Commit/tag | Notes |
|-------|--------|------------|-------|
| 1 Setup | in-progress | — | uv project, deps, ruff/mypy/pytest config, constants skeleton |
| 2 Data | todo | — | |
| 3 Fine-tuning + evaluation | todo | — | ends at STOP POINT 1 |
| 4 KV-cached decoding | todo | — | |
| 5 Attention | todo | — | |
| 6 Speculative decoding | todo | — | |
| 7 Application | todo | — | ends at STOP POINT 2 |
| 8 Long-running jobs | todo | — | user runs on Colab |
| 9 Report + docs + hygiene | todo | — | |

## Checklist state
### Phase 1
- [x] `uv init --package trialmatch`; Python 3.11.
- [x] `uv add torch transformers peft datasets accelerate numpy scikit-learn gradio pandas matplotlib huggingface_hub`; `uv add --dev pytest ruff mypy`.
- [x] Configure in `pyproject.toml`: ruff (format + lint incl. `D`, `I`, `ANN`, `UP`), mypy `strict = true`, pytest.
- [x] `constants.py` skeleton (paths, seeds, models, labels, training, prompt, decoding, benchmark, tolerances, app, `ADAPTER_REPO`).
- [ ] `PROGRESS.md`, `docs/DECISIONS.md`, README skeleton, `.gitignore`.
- [ ] Record dependency versions in `Facts discovered`.
- [ ] First commit + tag `phase-1-done`; add GitHub remote.

### Phase 2
- [ ] `AnnotationLoader.load()` drops every `gpt4_*` / `explanation_correctness` field.
- [ ] `SplitManager` 37/8/8 by patient, `splits/split_manifest.json`, `assert_no_leakage()`.
- [ ] `SentenceSplitter` (dataset numbering + regex fallback for app notes).
- [ ] `PromptBuilder` (system + numbered sentences + criterion + schema; compact JSON target).
- [ ] Truncation analysis -> `results/truncation_report.json`.
- [ ] `uv run python -m trialmatch.data --summary` runs.
- [ ] Self-review, commit, tag `phase-2-done`.

### Phase 3
- [ ] `ModelLoader` (base/adapter/merged, device+dtype).
- [ ] `FineTuner` (Trainer + peft LoRA, completion-only loss, `--resume`, `--output-dir`, `--smoke`).
- [ ] `results/train_log.json` + `report/figures/train_curves.pdf`.
- [ ] Optional `--sweep` (<=4 configs, validation macro-F1).
- [ ] `OutputParser` (strict + tolerant fallback, invalid counts as wrong label).
- [ ] `Evaluator` (classification overall/inclusion/exclusion, evidence micro/macro, empty-set rule).
- [ ] Base vs fine-tuned on val; test only via `--split test --final`.
- [ ] Error-analysis helper -> `results/error_cases.md` (>=12 cases).
- [ ] `OverallAggregator` documented rule.
- [ ] Colab jobs 1-3 in README + Awaiting user.
- [ ] Self-review, commit, tag `phase-3-done` -> **STOP POINT 1**.

### Phase 4
- [ ] `Decoder` ABC + `GenerationResult`.
- [ ] `UncachedDecoder` reference path.
- [ ] `CachedDecoder` explicit loop (no `model.generate`), documented masking/positions/cache math.
- [ ] Timing with CUDA sync/peak memory, warm-ups + repeats.
- [ ] `tests/test_decoding.py` greedy equality 3 prompt lengths x 2 output lengths.
- [ ] `Benchmarker` decoding grid -> `results/bench_decoding.json` + figure.
- [ ] Self-review, commit, tag `phase-4-done`.

### Phase 5
- [ ] `ReferenceAttention`.
- [ ] `TiledAttention` online softmax.
- [ ] Tests: shapes incl. non-multiples, causal/non-causal, fp32/fp16, vs reference + SDPA.
- [ ] Benchmark vs SDPA backends -> `results/bench_attention.json` + figure.
- [ ] Self-review, commit, tag `phase-5-done`.

### Phase 6
- [ ] `SpeculativeDecoder` (tokenizer equality assert, greedy accept/reject, cache cropping).
- [ ] Acceptance-rate and latency metrics.
- [ ] Tests: exact equality with greedy target for gamma in {1,3,5}.
- [ ] Benchmark -> `results/bench_speculative.json` + figure.
- [ ] Self-review, commit, tag `phase-6-done`.

### Phase 7
- [ ] `ScreeningPipeline.screen()`.
- [ ] `ReportExporter` (Markdown/HTML, disclaimer, provenance).
- [ ] Gradio app: disclaimer, screening tab, benchmark tab, error handling, lazy model load.
- [ ] README "Run the app".
- [ ] `--smoke` UI wiring test.
- [ ] Self-review, commit, tag `phase-7-done` -> **STOP POINT 2**.

### Phase 8
- [ ] Six Colab jobs documented in README + Awaiting user.

### Phase 9
- [ ] `report/main.tex` (IEEEtran, all sections, algorithm blocks).
- [ ] `refs.bib` >=5 verified real sources.
- [ ] Numbers only from `results/*.json`; `TODO(user)` where missing.
- [ ] `docs/DECISIONS.md`, `docs/AI_USE.md`, README complete.
- [ ] Final gate green, clean tree, no weights committed.

## Awaiting user (jobs only the user can run / things needed)
- [ ] Run Colab jobs 1-6 (listed in README "Long-running jobs (run on Colab T4)") — available after Phase 3/4/5/6/7 code lands.
- [ ] Compile `report/` on Overleaf; record 5-7 min video; take app screenshots.
- Results received from user: none yet.

## Facts discovered
- Environment: macOS 26.6.2 arm64, uv 0.12.5, Python 3.11 (uv-managed).
- Resolved versions: torch 2.15.0, transformers 5.22.1, peft 0.21.2, gradio 6.31.0,
  datasets 5.1.0, huggingface_hub 2.4.0, numpy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1,
  matplotlib 3.11.2, ruff 0.17.3, mypy 2.4.1, pytest 9.2.0.
- `transformers` 5.x and `gradio` 6.x are newer than the versions assumed by PLAN.md: every API
  call must be verified against the installed source before use.
- MPS is available locally; CPU/MPS only, no CUDA. No local GPU.

## Open problems / blockers
- None yet.

## Session log (append-only, newest last)
- 2026-11-04T20:15 opencode: Phase 1 started. Created uv project (root moved from
  `trialmatch/` subdir to repo root), added runtime+dev deps, wrote `pyproject.toml`
  tooling config, `.gitignore`, `constants.py` skeleton. Next: README, DECISIONS, first commit.
