# TrialMatch — criterion-level trial screening with a fine-tuned Qwen LoRA

> **Disclaimer.** Educational screening prototype, not a clinical enrollment decision.

Maps a patient note to trial eligibility criteria one criterion at a time: the model returns a
label (`included` / `not included` / `not enough information` / `not applicable`, or the exclusion
analogues) plus the patient sentence IDs that support it. Explanations are generated
deterministically from the label and the cited sentences — never by the model — and GPT-4
annotations from the source dataset are dropped at load time.

Built for GenAI Assignment #2 on `ncbi/TrialGPT-Criterion-Annotations` with
`Qwen/Qwen2.5-1.5B-Instruct` + LoRA, plus hand-written KV-cached decoding, tiled (online-softmax)
attention and speculative decoding.

## Repository map

```
src/trialmatch/     constants, data, schema, modeling, train, evaluate,
                    decoding, attention, speculative, benchmark, pipeline, app
tests/              fast CPU-only unit tests with tiny random models
splits/             frozen patient-level split manifest (seed 42)
results/            JSON/CSV/markdown outputs of the long runs (committed, small)
report/             IEEEtran paper, refs.bib, generated figures
docs/               ARCHITECTURE.md, DECISIONS.md, AI_USE.md
```

## Setup

```bash
uv sync                                    # creates .venv and installs from uv.lock
uv run python -c "import trialmatch; print(trialmatch.__version__)"
```

Python 3.11 is pinned in `.python-version`; `uv` fetches it automatically. No `pip` is used
anywhere except the single `!pip install uv` bootstrap on Colab.

## Quality gate

```bash
uv run ruff format . && uv run ruff check . && uv run mypy --strict src tests && uv run pytest -q
```

## Data and splits

```bash
uv run python -m trialmatch.data --summary
```

Writes `splits/split_manifest.json` (37/8/8 patients, seed 42) and
`results/truncation_report.json`. The test split is only ever evaluated through
`--split test --final`, once, at the end.

## Run the app

```bash
uv run python -m trialmatch.app            # http://127.0.0.1:7860
```

TODO(phase-7): first-load time, CPU/GPU notes, local-adapter instructions.

## Long-running jobs (run on Colab T4)

TODO(phase-3+): Colab setup cell, jobs 1-6 with commands, runtimes and outputs.

## Results

TODO(phase-9): table filled only from files in `results/`.

## Documentation

- `docs/ARCHITECTURE.md` — module/class map and data flow.
- `docs/DECISIONS.md` — alternatives considered/rejected, difficulties, R1-R10 audit.
- `docs/AI_USE.md` — AI-assistance appendix.
- `PROGRESS.md` — live status and handoff log.

## Licence and attribution

Dataset: TrialGPT criterion annotations (NCBI). Base model: Qwen2.5 (Apache-2.0). LoRA adapter:
`RayanKhan-79/genai-assignment-2` on the Hugging Face Hub.
