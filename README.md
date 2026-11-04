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
uv sync
uv run python -m trialmatch.app                                  # http://127.0.0.1:7860
uv run python -m trialmatch.app --device cpu                     # force CPU (CUDA -> MPS -> CPU)
uv run python -m trialmatch.app --adapter checkpoints/adapter    # explicit local adapter
```

- **First load is the slow part.** The fine-tuned 1.5B target (about 6 GB in fp32 on CPU) is loaded
  lazily on the first screening request: roughly 1-2 min on a CUDA T4 and 2-4 min on CPU; later
  requests reuse it. Nothing is downloaded until then.
- **Adapter resolution.** The app uses `--adapter`, else `checkpoints/adapter`, else downloads
  `RayanKhan-79/genai-assignment-2` into `checkpoints/adapter`. If none is reachable it runs the
  base model and says so in the status line.
- **Screening tab.** Paste a note (numbered lines are kept as-is; free text is split into
  sentences) or upload a `.txt`; add one inclusion and one exclusion criterion per line; choose
  `Uncached`, `KV cache` or `Speculative`; press Run. The table shows label, evidence IDs, quoted
  evidence and the deterministic template explanation, followed by the preliminary outcome and the
  missing-information list. **Download Markdown report** saves the report; an HTML copy is written
  next to it under `checkpoints/reports/`, with the disclaimer, aggregation rule and provenance
  block.
- **Benchmark tab.** Runs all three decoders on the first criterion of the current input and shows
  latency, tokens/s, forward calls, peak memory, acceptance rate (speculative) and whether the
  tokens match the uncached baseline, plus latency/throughput charts. The accordion below renders
  the stored `results/bench_*.json` files from Colab job 5.
- **Speculative mode** additionally loads `Qwen/Qwen2.5-0.5B-Instruct` as the draft (about 1 GB
  fp32) and asserts that both tokenizers are identical. On CPU it is expected to be slower than the
  KV cache; the benchmark tab reports the real number.
- **Errors** (empty note, no criteria, prompt over 2048 tokens, unparseable output, model not
  loadable) become a readable status message; the app stays usable and never shows a stack trace.
- `uv run python -m trialmatch.app --smoke` builds the whole UI and exercises both tabs on the
  0.5B base model without opening a server — the agent-side wiring check.

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

### Job 4 - test evaluation, exactly once (about 10-20 min)

Run this only after jobs 1 and 3 look right. The guard refuses the test split without `--final`, and
the report must never be tuned on it. The fine-tuned command also writes the error analysis, so the
test split is decoded only once.

```bash
uv run python -m trialmatch.evaluate --split test --model base --final
uv run python -m trialmatch.evaluate --split test --model finetuned \
    --adapter /content/drive/MyDrive/trialmatch/adapter --final --errors
```

Outputs: `results/test_base_hf.json`, `results/test_finetuned_hf.json`, their
`*_predictions.jsonl`, confusion matrices in `report/figures/`, and `results/error_cases.md`
(at least 12 failures with note, criterion, gold and prediction).

### Job 5 - benchmarks (about 10-25 min total, run after the Phase 6 code lands)

```bash
uv run python -m trialmatch.benchmark --suite decoding     # results/bench_decoding.json
uv run python -m trialmatch.benchmark --suite attention    # results/bench_attention.json
uv run python -m trialmatch.benchmark --suite speculative \
    --model finetuned --adapter /content/drive/MyDrive/trialmatch/adapter
```

Every suite prints the JSON path and writes its figure to `report/figures/`. The speculative suite
uses the fine-tuned 1.5B model as the verifier and `Qwen/Qwen2.5-0.5B-Instruct` as the draft
(identical tokenizer), decodes a fixed token budget on 8 validation prompts, and reports acceptance
rate, target/draft forward counts, latency and the speedup **or slowdown** against the cached
target; it also cross-checks one prompt with `generate(assistant_model=draft)`. Tune the load with
`--prompts N` and `--max-new-tokens N` (defaults 8 and 128). Optional: `train --model draft`
fine-tunes a draft adapter, but this benchmark uses the base draft. Optional equivalence check for
the hand-written cache (same prompt, same greedy rule, same parser):

```bash
uv run python -m trialmatch.evaluate --split val --model finetuned \
    --adapter /content/drive/MyDrive/trialmatch/adapter --decode cached
```

### Job 6 - upload the adapter to the Hugging Face Hub (about 2 min)

```bash
uv run hf auth login                      # or export HF_TOKEN from Colab secrets
uv run hf upload RayanKhan-79/genai-assignment-2 \
    /content/drive/MyDrive/trialmatch/adapter
```

The repository must match `constants.ADAPTER_REPO` (`RayanKhan-79/genai-assignment-2`); the app
downloads it there when no local adapter is present. Model page:
<https://huggingface.co/RayanKhan-79/genai-assignment-2>.

### Local wiring checks (agent-side, never committed as results)

```bash
uv run python -m trialmatch.train --smoke                       # 2 steps, 8 examples, 0.5B, CPU
uv run python -m trialmatch.evaluate --split val --model base --smoke   # 4 records, 0.5B
uv run python -m trialmatch.app --smoke                         # both UI tabs, 0.5B, no server
```

## Results

Every number in the report comes from a file listed here; the table stays `TODO(user)` until the
corresponding Colab job has run and the file is committed.

| Item | Backing file | Status |
|------|--------------|--------|
| Fine-tuning log and curves | `results/train_log.json`, `report/figures/train_curves.pdf` | TODO(user) job 1 |
| Validation, base vs fine-tuned | `results/val_base_hf.json`, `results/val_finetuned_hf.json` | TODO(user) job 3 |
| Test, base vs fine-tuned (once) | `results/test_*.json`, `results/error_cases.md` | TODO(user) job 4 |
| Decoding benchmark | `results/bench_decoding.json` | TODO(user) job 5 |
| Attention benchmark | `results/bench_attention.json` | TODO(user) job 5 |
| Speculative benchmark | `results/bench_speculative.json` | TODO(user) job 5 |
| Adapter | <https://huggingface.co/RayanKhan-79/genai-assignment-2> | TODO(user) job 6 |
| Demo video (5-7 min) | TODO(user) link | TODO(user) |
| App screenshots | `report/figures/app_*.png` | TODO(user) |

The split manifest is `splits/split_manifest.json` (37/8/8 patients, seed 42) and the truncation
analysis is `results/truncation_report.json`.

## Documentation

- `docs/ARCHITECTURE.md` — module/class map and data flow.
- `docs/DECISIONS.md` — alternatives considered/rejected, difficulties, R1-R10 audit.
- `docs/AI_USE.md` — AI-assistance appendix.
- `PROGRESS.md` — live status and handoff log.

## Licence and attribution

Dataset: TrialGPT criterion annotations (NCBI). Base model: Qwen2.5 (Apache-2.0). LoRA adapter:
`RayanKhan-79/genai-assignment-2` on the Hugging Face Hub.
