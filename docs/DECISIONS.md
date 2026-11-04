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
