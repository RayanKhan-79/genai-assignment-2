# AGENTS.md

Project: TrialGPT criterion-matching system (GenAI Assignment #2, deadline 2026-11-10). Full spec in `PLAN.md`; live status in `PROGRESS.md`.

## Every session, first
1. Read `PLAN.md` sections A, 0 and 1 (find sections with `grep -n "^## " PLAN.md`), then `PROGRESS.md`. Read only the phase you are working on afterwards.
2. `git log --oneline -20`, `git status`; run the quality gate:
   `uv run ruff format . && uv run ruff check . && uv run mypy --strict src tests && uv run pytest -q`
3. Continue from `PROGRESS.md -> NEXT ACTION`. Do not redo finished work.

## Hard rules (never break)
- Use `uv` only (`uv add`, `uv run`); never `pip` or bare `python`.
- Type annotations everywhere (`mypy --strict` clean).
- Minimal global variables; no `global`, no mutable module state.
- All constants live in `src/trialmatch/constants.py`; import from there.
- Don't optimize prematurely. Write minimal code: 50 lines beat 1000.
- Write documentation (Google docstrings, README, docs/).
- Never run jobs longer than ~10 minutes (training, full evals, full benchmarks). Smoke tests only (`--smoke`); put the exact commands in README "Long-running jobs (run on Colab)" and in `PROGRESS.md -> Awaiting user`.
- One consistent style (ruff format/check config in `pyproject.toml`).
- Use classes (thin; config as frozen dataclasses; `main()` only instantiates).
- Test split only via `--final`; never use `gpt4_*` dataset fields; never put secrets in the repo; never claim results not backed by files in `results/`.

## Workflow
- Work autonomously phase by phase; at each phase end: self-review, quality gate, update `PROGRESS.md`, commit, tag `phase-N-done`, continue.
- Update `PROGRESS.md` after every meaningful step (you may be cut off any time; another agent will resume from it).
- Pause only at STOP POINT 1 (end of Phase 3) and STOP POINT 2 (end of Phase 7), as defined in `PLAN.md` section A.2.2, or if a user-only value (HF username, GitHub URL, author details) is missing.
- Don't rely on memory for library APIs (transformers cache classes, peft, gradio change often): read the installed source and verify with tiny-model tests.
- Ambiguity: choose the simplest option consistent with the assignment, log it in `docs/DECISIONS.md`, don't ask.
