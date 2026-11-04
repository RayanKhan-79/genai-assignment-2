# Architecture

One Python package (`src/trialmatch`) plus a Gradio entry point. Every module is importable without
side effects; `main()` functions only parse arguments and instantiate classes. All configuration and
constants live in `constants.py`.

## Module map

| Module | Public classes / functions | Responsibility |
|--------|---------------------------|----------------|
| `constants.py` | frozen dataclasses `TrainConfig`, `GenerationConfig`, `BenchmarkConfig`; paths, labels, prompt templates, app strings | single place for every constant (rule R4) |
| `data.py` | `AnnotationLoader`, `Annotation`, `Example`, `SplitManager`, `SplitManifest`, `SentenceSplitter`, `PromptBuilder`, `PromptEncoder`, `TruncationAnalyzer` | load the CSV-style dataset, drop every `gpt4_*` field, freeze a patient-level split, render prompts and tokenise with completion-only labels |
| `schema.py` | `Prediction`, `OutputParser` | strict JSON parse -> regex recovery -> `invalid`; evidence IDs are range-checked against the note |
| `modeling.py` | `LoadedModel`, `ModelLoader` | load base weights, optionally apply/merge a PEFT adapter, pick CUDA/MPS/CPU and fp16/fp32 |
| `train.py` | `FineTuner`, `Sweeper`, `main` | LoRA training through `Trainer` with a completion-only collator; `--smoke`, `--resume`, `--sweep` |
| `evaluate.py` | `EvalRecord`, `GenerationRunner`, `MetricsCalculator`, `ErrorAnalyzer`, `Evaluator`, `main` | greedy generation (`hf`, `uncached`, `cached`, `speculative`), classification/evidence metrics, confusion matrices, error cases |
| `decoding.py` | `GenerationResult`, `Timing`, `Decoder`, `UncachedDecoder`, `CachedDecoder`, `DecoderFactory` | hand-written greedy loops; the cached loop feeds one token per step into a `DynamicCache` |
| `attention.py` | `ReferenceAttention`, `TiledAttention` | full-matrix reference and blocked online-softmax attention |
| `speculative.py` | `SpeculativeStats`, `SpeculativeDecoder` | draft/verify greedy loop with cache cropping and tokenizer equality assertion |
| `benchmark.py` | `Benchmarker`, `DecodingBenchmark`, `AttentionBenchmark`, `SpeculativeBenchmark`, plotters, `main` | the three benchmark suites; `--smoke` builds tiny random models and writes under `checkpoints/smoke/` |
| `pipeline.py` | `CriterionSpec`, `Provenance`, `ScreeningItem`, `ScreeningResult`, `ScreeningPipeline`, `ReportExporter`, `OverallAggregator` | per-criterion screening, deterministic explanations, Markdown/HTML reports and the documented outcome rule |
| `app.py` | `ScreeningApp`, `main` | Gradio Blocks UI with lazy model loading, screening tab, live benchmark tab and stored-JSON viewer |

## Data flow

```
dataset (HF)                                        results/ (committed)
   | AnnotationLoader                                   ^
   v                                                    |
Annotation -> SplitManager -> splits/split_manifest.json|
   |                                                    |
   v                                                    |
PromptBuilder -> Example -> PromptEncoder -> token ids  |
                              |                         |
                              v                         |
                    model / decoder -> raw text -> OutputParser -> Prediction
                              |                                    |
                              v                                    |
                        Evaluator -> metrics, predictions.jsonl, figures
                              |
                              v
                    ScreeningPipeline.assemble -> ScreeningResult -> ReportExporter
                              |
                              v
                        Gradio app (screen + benchmark tabs)
```

Training follows the same path up to `PromptEncoder.encode_train`, which masks prompt tokens so the
loss covers only the JSON completion.

## Decoding invariant

`UncachedDecoder` recomputes the whole sequence each step (`use_cache=False`). `CachedDecoder`
prefills once, then feeds exactly one token per step with `position_ids = [[cache_len]]` and an
attention mask of length `cache_len + 1`. Both are greedy, so they agree token-for-token in fp32
(tests) and are compared on the real model in the decoding benchmark.

`SpeculativeDecoder` keeps two caches at the same prefix. Each round drafts `gamma` tokens, verifies
`[pending] + proposals` with one target forward, then crops the target by the rejected tail and the
draft by one fewer position (consuming the last proposal when everything matched). The emitted
sequence is exactly the target's greedy sequence; this is tested against `CachedDecoder` and
cross-checked on the real model with `generate(assistant_model=draft)`.

## Attention invariant

`ReferenceAttention` forms the full `T x T` matrix and calls `torch.softmax`. `TiledAttention`
iterates query blocks and key blocks, keeping a running maximum, normaliser and accumulator per row,
and rescales the accumulator whenever the maximum changes. Fully masked rows are guarded so they
return zeros, not NaNs.

## Application flow

`ScreeningApp` holds the loaded target and draft models plus one cached `ScreeningPipeline` per
decoding mode. The first request resolves the adapter (`--adapter`, then `checkpoints/adapter`, then
`snapshot_download(ADAPTER_REPO)`), loads a merged model and reuses it. Screening writes a Markdown
and an HTML report under `checkpoints/reports/` (never `results/`) and returns the Markdown path to
the download button. The benchmark tab times the same prompt with all three decoders and renders the
stored `results/bench_*.json` files from the Colab run.
