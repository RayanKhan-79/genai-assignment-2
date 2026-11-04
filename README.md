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

Nothing in this section may be run locally: each job takes longer than 10 minutes on the target
task (rule R8). Only `--smoke` variants are run by the agent.

### Colab setup cell (once per session)

```python
# Runtime -> Change runtime type -> T4 GPU
!pip install uv                      # the only allowed pip use: it bootstraps uv
!git clone https://github.com/RayanKhan-79/genai-assignment-2
%cd genai-assignment-2
!uv sync
from google.colab import drive
drive.mount("/content/drive")
# Colab secrets (key icon): add HF_TOKEN and a GitHub token, enable notebook access
```

Push after every job so the work survives a disconnect:

```bash
git add results report/figures splits && git commit -m "results: job <n>" && git push
```

### Job 1 - LoRA fine-tuning (about 20-40 min on T4)

```bash
uv run python -m trialmatch.train --output-dir /content/drive/MyDrive/trialmatch/adapter
# after a disconnect, resume the same command with:
uv run python -m trialmatch.train --output-dir /content/drive/MyDrive/trialmatch/adapter --resume
```

Outputs: `results/train_log.json`, `report/figures/train_curves.pdf`, adapter in the Drive folder
(`adapter_model.safetensors`, `adapter_config.json`, tokenizer files).

### Job 2 - optional hyper-parameter sweep (about 1.5-2.5 h)

```bash
uv run python -m trialmatch.train --sweep
```

Four configurations (LoRA rank x learning rate) selected by **validation macro-F1 only**.
Output: `results/sweep.json`; the best adapter stays in `checkpoints/sweep/`.

### Job 3 - validation evaluation, base vs fine-tuned (about 10-20 min each)

```bash
uv run python -m trialmatch.evaluate --split val --model base
uv run python -m trialmatch.evaluate --split val --model finetuned \
    --adapter /content/drive/MyDrive/trialmatch/adapter
```

Outputs: `results/val_base_hf.json`, `results/val_finetuned_hf.json`, the matching
`*_predictions.jsonl`, and confusion matrices in `report/figures/`.

### Job 5 - benchmarks (about 10-25 min total, run after the Phase 6 code lands)

```bash
uv run python -m trialmatch.benchmark --suite decoding     # results/bench_decoding.json
uv run python -m trialmatch.benchmark --suite attention    # results/bench_attention.json
```

Every suite prints the JSON path and writes its figure to `report/figures/`. Optional equivalence
check for the hand-written cache (same prompt, same greedy rule, same parser):

```bash
uv run python -m trialmatch.evaluate --split val --model finetuned \
    --adapter /content/drive/MyDrive/trialmatch/adapter --decode cached
```

### Local wiring checks (agent-side, never committed as results)

```bash
uv run python -m trialmatch.train --smoke                       # 2 steps, 8 examples, 0.5B, CPU
uv run python -m trialmatch.evaluate --split val --model base --smoke   # 4 records, 0.5B
```

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
