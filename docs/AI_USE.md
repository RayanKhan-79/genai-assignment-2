# AI-use appendix

This project was developed with an AI coding agent under a written working agreement
(`AGENTS.md` / `PLAN.md`). The account below is deliberately specific about what the agent did and
what the human author did, and about how every AI-generated claim is checked.

## Agents and models used

- **Primary coding agent:** OpenCode running the model `DeepSeek V4.1 Flash Free` (session log in
  `PROGRESS.md` records the agent/model string per update).
- **Tools available to the agent:** shell, file edits, repository search, the project's own quality
  gate (`ruff`, `mypy --strict`, `pytest`), local smoke runs, and read-only web fetches for library
  and citation verification.
- **No other model was used to generate results.** The fine-tuned model is Qwen2.5-1.5B-Instruct
  with a LoRA adapter, trained and evaluated by scripts in this repository.

## What the agent did

- Translated the specification in `PLAN.md` into the module layout, classes, tests and docs listed in
  `docs/ARCHITECTURE.md`.
- Verified library APIs against the installed source (transformers 5.22, peft 0.21, gradio 6.31,
  torch 2.15) instead of relying on memory; the decisions log records the resulting changes
  (for example `warmup_steps` instead of `warmup_ratio`, `Cache.crop(tokens_to_remove)` semantics,
  and the Gradio 6 `DownloadButton`/`Plot` output contracts).
- Wrote the unit tests, including the exact-equivalence tests for cached/uncached decoding
  (`tests/test_decoding.py`), tiled versus reference attention (`tests/test_attention.py`) and
  speculative versus cached target decoding (`tests/test_speculative.py`).
- Ran only local smoke checks on tiny random or 0.5B models; long jobs were left to the human.
- Drafted the paper skeleton and the documentation in this directory.

## What the human author did

- Chose the project requirements and supplied the HF repository, GitHub repository and author
  details.
- Ran every long job on Colab (training, validation/test evaluation, benchmarks, adapter upload),
  pushed the outputs, compiled the report on Overleaf and recorded the demo video.
- Reviewed and owns every architectural decision, every reported number and every claim in the
  paper. The paper contains no result that is not traceable to a file in `results/`.

## How AI output was verified

1. **Quality gate on every phase:** `uv run ruff format . && uv run ruff check . &&
   uv run mypy --strict src tests && uv run pytest -q`. Phase boundaries and their gate results are
   recorded in `PROGRESS.md`.
2. **Tiny-model equivalence tests** rather than "looks right": the cached decoder must produce the
   same tokens as the uncached reference, tiled attention must match the full reference within
   fp32/fp16 tolerances, and speculative decoding must reproduce the target's greedy sequence for
   several `gamma` values and prompt lengths.
3. **Smoke runs** exercise the real pipeline end to end (2 training steps, 4-record evaluation,
   both UI tabs) and write under `checkpoints/smoke/` so they can never be mistaken for results.
4. **Citation checks:** every entry in `report/refs.bib` was checked against its arXiv or publisher
   page during this session; no citation was invented.
5. **Human review at the two stop points** defined in `PLAN.md`, before the long jobs and before
   submission.

## Risks and limitations

- AI-generated code can be subtly wrong even with green tests; the equivalence tests target the
  places where this is most likely (cache positions, masking, speculative cache re-sync), but they
  are not a proof.
- The report skeleton originally contained explicit `TODO(user)` markers. Every one was replaced
  with a value backed by `results/` before submission; the human author supplied the screenshots and
  the video link, and compiled the final PDF on Overleaf.
- Any AI suggestion that could not be verified was either removed or logged as a decision in
  `docs/DECISIONS.md` rather than silently kept.
