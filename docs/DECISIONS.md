# Decisions log

Append-only record of decisions that are **not** already fixed by `PLAN.md`, plus the
alternatives considered and the difficulties met (rubric A3). Rule R1-R10 audit lives at the
bottom of this file.

## D1. Adapter repository name — 2026-11-04
The user supplied `https://huggingface.co/RayanKhan-79/genai-assignment-2`, whereas `PLAN.md`
uses the template `<HF_USERNAME>/trialmatch-qwen-lora`. The user-supplied value wins:
`constants.ADAPTER_REPO = "RayanKhan-79/genai-assignment-2"`. The repo must be created/updated by
the user (job 6) and the agent never handles the token.

## D2. GitHub remote — 2026-11-04
Remote is `https://github.com/RayanKhan-79/genai-assignment-2` (`origin`). Author for the report:
Rayan Khan (email `rayankhannexus@gmail.com` from the local git config). Pushes are attempted only
when credentials work locally; otherwise the user pushes from Colab or their machine.

## D3. Repository root — 2026-11-04
`uv init --package` created a nested `trialmatch/` directory; its contents were moved to the
repository root so that the layout matches `PLAN.md` section 2 exactly
(`src/trialmatch/`, `tests/`, `results/`, ...). The `[project.scripts]` entry that `uv init`
generated was deleted: every entry point is `python -m trialmatch.<module>`, which removes code.

## D4. Dependency versions — 2026-11-04
`uv` resolved much newer libraries than `PLAN.md` assumed: `transformers` 5.22, `gradio` 6.31,
`peft` 0.21, `torch` 2.15, `datasets` 5.1, `huggingface_hub` 2.4. Pinning to old versions was
rejected (the Colab image installs from `uv.lock`, so a newer, still-supported stack is preferable).
Consequence: every `transformers` cache/Trainer/generate API and every `gradio` component signature
is verified against the installed source (and tiny-model tests) instead of being written from memory.

## D5. Local compute — 2026-11-04
The agent machine is an Apple-silicon Mac (CPU/MPS, no CUDA). All agent-run checks are CPU-only and
tiny (`--smoke`); real GPU runs happen on Colab T4 by the user (R8). `ModelLoader` therefore selects
CUDA -> MPS -> CPU automatically and uses fp16 only on GPU.

## D6. Labels are never collapsed — 2026-11-04
Inclusion and exclusion keep their four categories each. `invalid` (unparseable output) is an extra
predicted value used only for scoring, never a training target.

## R1-R10 audit
Filled at the end of Phase 9; each phase adds evidence below.

| Rule | Evidence |
|------|----------|
| R1 uv only | `uv.lock` + `pyproject.toml` committed; README lists only `uv` commands; only allowed `pip` use is `!pip install uv` in the Colab bootstrap cell. |
| R2 annotations | `uv run mypy --strict src tests` green (Phase 1). |
| R3 no globals | `grep -rn "\bglobal " src` empty; no mutable module state. |
| R4 constants | `src/trialmatch/constants.py`; phased grep for literals. |
| R5 no premature optimisation | baseline implementations first; any optimisation cites a measurement. |
| R6 minimal code | `src/` kept small; library calls preferred except the three hand-written algorithms. |
| R7 docs | Google docstrings everywhere, `ruff check --select D`, README + `docs/`. |
| R8 no long jobs | only `--smoke` runs locally; six Colab jobs documented. |
| R9 style | `ruff format` + `ruff check` clean. |
| R10 classes | thin classes, frozen dataclass configs, `main()` only instantiates. |

## D7. Prompt wording (Phase 3) — 2026-11-04
The first template ended with `Use [] for evidence_ids when no sentence supports the label.`.
The 0.5B base model copied that literal `[]` and every completion was unparseable (invalid rate
1.0 in the smoke run). The instruction was reworded to *describe* the schema instead of showing a
bare `[]`; the same smoke run then produced strict JSON for 4/4 records. No few-shot examples are
used, so the before/after comparison keeps the identical prompt for both models.

## D8. Labels in metrics (Phase 3) — 2026-11-04
`overall` classification uses the six distinct labels across both criterion types; `inclusion` and
`exclusion` use their own four. Invalid output is never added as a class: it scores as a miss for
the gold class (accuracy and recall drop, precision is unaffected). sklearn's
`classification_report` omits `accuracy` for degenerate batches, so accuracy is computed explicitly
and stored next to the report.

## D9. Local smoke runs use the CPU (Phase 3) — 2026-11-04
The agent machine is an Apple-silicon Mac with ~9 GiB of MPS memory; a 0.5B fp32 backward pass plus
the 151k-vocabulary logits exceeds it (`MPS backend out of memory`). `--smoke` therefore forces
`use_cpu=True` for `Trainer` and batch size 1, and keeps its artefacts inside `checkpoints/smoke/`
instead of `results/`. Smoke outputs are deleted afterwards so `results/` only ever holds real runs.

## D10. Extra test modules — 2026-11-04
PLAN.md section 2 lists five test files. Two more were added because the corresponding code is pure
logic worth unit-testing without a model: `tests/test_evaluate.py` (metric/empty-set rules) and
`tests/test_pipeline.py` (aggregation rule), plus `tests/test_constants.py` (label-set invariants).

## D11. Source-size budget (R6) — 2026-11-04
PLAN.md suggests "roughly under 1,200 lines" for `src/`. After Phase 3 the package is 1,742 lines
across seven modules, and Phases 4-7 still add three hand-written algorithms, a benchmarker and the
app. Every module is required by the specification (no dead code, no speculative abstraction), so
the budget is treated as a guideline and the final figure is reported in the R1-R10 audit instead of
being met by deleting required features.

## D12. No `device_map="auto"` — 2026-11-04
The target model in fp16 is ~3 GB and the T4 has 16 GB, so accelerate sharding buys nothing;
loading the whole checkpoint and calling `.to(device)` keeps the Trainer path simple and avoids
`device_map` interactions (which are meant for multi-GPU / quantised loading). Documented deviation
from the PLAN.md wording; behaviour on CPU/MPS is identical.

## D13. Evaluation decoding mode (Phase 4) — 2026-11-05
`evaluate` defaults to `--decode hf` (`model.generate`, greedy) so the job-3 commands already handed
to the user stay reproducible. The hand-written decoders are selected explicitly with
`--decode cached|uncached`, which writes `results/{split}_{model}_cached.json`; running that on
validation is the equivalence check for the cache (same prompt, same greedy rule, same parser).
Result filenames always carry the decoding mode, so no two runs can be confused.

## D14. Smoke artefacts live outside `results/` (Phase 4) — 2026-11-05
Benchmarks and evaluations run with `--smoke` write under `checkpoints/smoke/` and tag their files
`smoke-<mode>`; `results/` only ever holds real Colab runs, so every number quoted in the report is
traceable to a real job.

## D15. Attention benchmark shapes and ground truth (Phase 5) — 2026-11-05
The suite uses batch 1, 8 heads, head_dim 64 and causal masking (the shape of one decode step at a
realistic width). Errors are computed against `ReferenceAttention` in **fp32**, so the reference row
is exactly zero by construction and every other variant is measured against the mathematical
definition rather than against itself. Unsupported SDPA backends are recorded as
`{"supported": false}` instead of being silently skipped, so the T4 report can state which backends
(typically FlashAttention) are unavailable.

## D16. Speculative decoding benchmark protocol (Phase 6) — 2026-11-05
- **Fixed budget, no EOS stop.** The benchmark decodes `max_new_tokens` (default 128) on every prompt
  for the cached baseline and every gamma, so throughput compares the same amount of token work.
  Losslessness is checked per prompt against the cached target's full sequence; any divergence is
  recorded (`tokens_equal`, `mismatched_prompts`) instead of being averaged away.
- **Models.** Target = fine-tuned 1.5B with the adapter merged (needed for the hand-written cache);
  draft = `Qwen/Qwen2.5-0.5B-Instruct`, whose tokenizer is asserted identical (vocab + special
  tokens). The optional draft LoRA from PLAN is not used: acceptance rate is already measurable with
  the base draft and the extra Colab job was not worth the time budget.
- **Load.** 8 validation prompts (`--prompts`), gammas 2/4/6 (`BenchmarkConfig.speculative_gammas`).
  Each decoder warms up on the first prompt `warmup_runs` times, then every prompt is timed once;
  mean/std are across prompts, which keeps the Colab job to a few minutes while n=8 is enough for a
  clear speedup/slowdown statement.
- **Independent reference.** One prompt is also decoded with Hugging Face's
  `generate(assistant_model=draft)`. HF stops at EOS while the benchmark does not, so the check
  compares the common prefix and reports both lengths; a prefix mismatch means the hand-written
  loop is not greedy.
- **Statistics field.** `GenerationResult.stats` is optional and typed through a `TYPE_CHECKING`
  import so `decoding.py` does not import `speculative.py` at runtime (no cycle).

## D17. Application and report choices (Phase 7) — 2026-11-05
- **One merged target model for every decoder.** The app always loads with `merge_adapter=True`, so
  `uncached`, `cached` and `speculative` all see identical weights; an unmerged PEFT model would
  change the forward pass and break the decoder-comparison tab.
- **Criteria input.** Two textboxes (one criterion per line) instead of a test-split example
  dropdown: PLAN allows either, and this needs no dataset download at app start.
- **Reports.** `ReportExporter.write` emits Markdown (offered by the Download button) and HTML (same
  stem) under `checkpoints/reports/` — app artefacts never enter `results/`, so every committed
  result still comes from a named Colab job.
- **Benchmark tab.** It times all three decoders on the **first** criterion of the current input
  with at most 32 new tokens (`APP_BENCHMARK_MAX_NEW_TOKENS`) and speculative gamma 4, which keeps
  the UI responsive on CPU; the honest speedup/slowdown study is job 5 on the T4.
- **Speculative in the app** loads the 0.5B draft lazily too and asserts tokenizer equality; on CPU
  it is expected to be slower, and the tab reports that instead of hiding it.
- **`--smoke` does not launch a server.** It builds the Blocks layout (validates component wiring),
  runs both tabs once on the 0.5B base model and exits, because the agent environment has no browser
  and a blocking server would violate the 10-minute rule.
- **Typing.** Gradio 6 has no complete stubs, so `trialmatch.app` alone gets
  `disallow_untyped_calls = false`; every other module stays under full `mypy --strict`.
- **Runner reuse.** `GenerationRunner` gained an optional `draft` plus a `speculative` mode, so the
  app, the pipeline and the evaluator share one decoder path; the evaluator's `--decode` choices
  are unchanged.

## D18. Report skeleton and job 4 sequencing (Phase 9, results-independent) — 2026-11-05
- **No invented numbers.** `report/main.tex` contains only facts backed by committed files (dataset
  counts, split sizes, hyper-parameters, token limits) and uses a red `\pending{...}` macro for every
  result cell. The paper must not compile with a fabricated number.
- **Compiles before the figures exist.** Every `\includegraphics` goes through `\pendingfigure`,
  which falls back to a labelled box when the PDF/PNG is missing. This lets the skeleton be checked
  on Overleaf immediately and keeps the real figures as drop-in replacements.
- **Citations.** Eight sources were verified against arXiv abstract pages or the publisher DOI
  (TrialGPT, Qwen2.5, LoRA, FlashAttention, online softmax, Leviathan et al., Chen et al., Vaswani
  et al.); no citation was written from memory.
- **Test split stays single-pass.** Job 4 attaches `--errors` to the fine-tuned `--final` run, so the
  test split is decoded exactly once while still producing `results/error_cases.md`; there is no
  second test invocation just for the error analysis.
- **README results table** lists the exact backing file for every number and marks it `TODO(user)`
  until the job has run, which is the checklist the final audit will use.

## D19. Headless matplotlib and non-fatal plotting (job-1 bugfix) — 2026-11-05
- **Symptom.** On Colab, job 1 trained all three epochs and saved the adapter and
  `results/train_log.json`, then died in `TrainCurvePlotter.plot` with
  `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value`.
- **Cause.** Colab exports `MPLBACKEND=module://matplotlib_inline.backend_inline`. That Python module
  lives in Colab's system environment, not in the `uv` venv where the job runs, so matplotlib cannot
  resolve the backend at import time. `matplotlib.use("Agg")` is too late: the failure happens on
  `import matplotlib`.
- **Fix.** Set `os.environ["MPLBACKEND"] = constants.MPL_BACKEND` (`"Agg"`) *before* importing
  matplotlib in every plotting path (training curve, confusion matrices, three benchmark suites, app
  figure). Alternative rejected: adding `matplotlib-inline` as a dependency — it is a notebook
  integration, not something a saved PDF needs.
- **No silent loss.** `_write_log` writes the JSON first and wraps the plot in `try/except`: the
  adapter and log are already on disk, so a drawing failure prints a warning plus the exact
  regeneration command instead of aborting the job. `train --plot-results <json>` redraws the figure
  without retraining, which is what job 1 needs.
- **Test.** `tests/test_train.py` runs the plotter in a subprocess with the invalid Colab value and
  asserts a `%PDF` file is produced, so the regression cannot come back unnoticed.

## D20. `Cache.crop` takes a negative count; forced-rejection tests (job-5 bugfix) — 2026-11-08
- **Symptom.** The speculative benchmark crashed on its first warm-up round with
  `ValueError: Calling crop with a positive value (2) is not supported`.
- **Cause.** transformers 5 changed `DynamicLayer.crop`: a negative argument removes that many tokens,
  a positive argument raises (the old behaviour treated a positive value as a target absolute size).
  Our loop passed `budget - matched`, which is positive whenever the draft is rejected.
- **Why tests missed it.** With tiny random Qwen2 models both target and draft collapse to one
  constant token, so every proposal matched and only `crop(0)` ran; the lossless test therefore
  never covered the rejection path. This is exactly the kind of gap the plan warns about.
- **Fix.** `target_cache.crop(matched - budget)` and `draft_cache.crop(-draft_drop)`; the docstring
  now states the sign convention. The original cache invariant is unchanged.
- **Tests.** A `ForcedDraft` wrapper overwrites logits (one scripted token per single-token step,
  prefill excluded) while still using the real KV cache. `test_rejected_proposals_are_cropped` forces
  every proposal to be rejected and asserts exact target equality with zero acceptance;
  `test_partial_acceptance_matches_target` mixes accepted and rejected proposals and asserts exact
  equality with `0 < accepted < drafted`. Both fail on the old sign, so the regression is covered.

## D21. Benchmark figures belong in `report/figures` (packaging bug) — 2026-11-08
- **Symptom.** Job 5 wrote `bench_*.pdf` next to `bench_*.json` in `results/`, while the report includes
  `figures/bench_*.pdf` and the README promised `report/figures/`.
- **Cause.** `Benchmarker._figure_root()` served as the root for *both* the JSON and the figure, so
  the real-run branch could only be `results/`.
- **Fix.** Split it into `_results_root()` (JSON -> `results/`) and `_figure_root()` (figures ->
  `report/figures/`); smoke runs still keep both under `checkpoints/smoke/`. The three job-5 PDFs were
  moved with `git mv`; no benchmark was re-run.

## D22. Job-1 log reconstruction and adapter provenance — 2026-11-08
- **Situation.** The job-1 Colab session died after training and before `results/train_log.json` could
  be pushed. The Drive adapter and the evaluations survived; only the log/curve were lost. The console
  output of the final training cell was preserved in the notebook and is reproduced in the session
  record.
- **Decision.** Rebuild `results/train_log.json` by transcribing that console output exactly (loss,
  grad-norm, learning rate, epoch, eval loss/runtime). Step numbers are derived deterministically,
  not invented: 38 steps/epoch (603 rows / (batch 2 x accumulation 8), rounded up) and
  `logging_steps = 5`, so train logs are at steps 5..35, 40..70, 80..110 and epoch-end evals at
  38/76/114. The `0.1325` epoch value in the output confirms step 5 (`5/38 = 0.1316`), and `1.053`
  confirms step 40 (`40/38 = 1.0526`).
- **Why not retrain.** Re-running job 1 would produce a different adapter (the console output shows a
  second run already differs from the first crashed one) and would silently invalidate jobs 3-5,
  which are the primary results. Transcription keeps the evaluated adapter and the reported curve
  tied to the same run.
- **Provenance.** `PROGRESS.md` and the README results table both mark the log as reconstructed.
  Epoch losses in the report were corrected to this run: 0.1349, 0.1170, 0.1227; train loss 0.1022;
  runtime 697 s.

## D23. Force `MPLBACKEND=Agg` before Gradio imports matplotlib — 2026-11-09
- **Symptom.** The first app request in Colab died in Gradio's queue with
  `ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value`.
- **Cause.** Colab exports that backend module, but `matplotlib-inline` is not installed in the `uv`
  environment. Gradio wraps every request in `MatplotlibBackendMananger`, which imports matplotlib
  and inherits `MPLBACKEND`; the plot helpers only fixed the variable when they ran, i.e. too late.
  The exception is raised in the queue worker, so the UI shows no result.
- **Fix.** `_force_agg_backend()` sets `MPLBACKEND = constants.MPL_BACKEND` at the start of
  `ScreeningApp.build()`, before the server accepts requests. `_benchmark_figure` reuses it.
- **Test.** `test_colab_backend_value_is_replaced` sets the Colab value, calls the helper and asserts
  `Agg`. Verified locally that importing Gradio does not import matplotlib, so `build()` is early
  enough.
