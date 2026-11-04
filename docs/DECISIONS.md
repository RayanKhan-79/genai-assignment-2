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
