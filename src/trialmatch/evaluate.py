"""Evaluation: greedy generation, classification/evidence metrics, figures and error analysis.

The test split is evaluated exactly once, through ``--split test --final``; the guard lives in
:meth:`trialmatch.data.SplitManager.assert_split_allowed`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from trialmatch import constants as c
from trialmatch.data import (
    Annotation,
    AnnotationLoader,
    Example,
    PromptBuilder,
    PromptEncoder,
    SentenceSplitter,
    SplitManager,
)
from trialmatch.decoding import DecoderFactory
from trialmatch.modeling import LoadedModel, ModelLoader
from trialmatch.schema import OutputParser, Prediction


@dataclass(frozen=True)
class EvalRecord:
    """One evaluated criterion: gold label/evidence, prediction, and its raw text."""

    annotation_id: int
    patient_id: str
    trial_id: str
    criterion_type: str
    criterion_text: str
    note: str
    gold_label: str
    pred_label: str
    gold_evidence: tuple[int, ...]
    pred_evidence: tuple[int, ...]
    raw: str
    strict: bool
    truncated_ids: tuple[int, ...]
    latency_s: float

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (tuples become lists)."""
        return {
            **asdict(self),
            "gold_evidence": list(self.gold_evidence),
            "pred_evidence": list(self.pred_evidence),
            "truncated_ids": list(self.truncated_ids),
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> EvalRecord:
        """Rebuild a record from :meth:`to_json` output."""
        return cls(
            **{
                **row,
                "gold_evidence": tuple(row["gold_evidence"]),
                "pred_evidence": tuple(row["pred_evidence"]),
                "truncated_ids": tuple(row["truncated_ids"]),
            }
        )


class GenerationRunner:
    """Greedy generation through ``model.generate`` or one of the hand-written decoders."""

    def __init__(
        self,
        loaded: LoadedModel,
        max_new_tokens: int = c.GENERATION.max_new_tokens,
        decode: str = "hf",
    ) -> None:
        """Store the model bundle, the generation budget and the decoding mode."""
        self._loaded = loaded
        self._encoder = PromptEncoder(loaded.tokenizer)
        self._max_new_tokens = max_new_tokens
        self._decode = decode
        eos = loaded.tokenizer.eos_token_id
        self._decoder = (
            None
            if decode == "hf"
            else DecoderFactory.create(
                decode, loaded.model, [eos] if eos is not None else [], loaded.device
            )
        )

    def run(self, example: Example) -> tuple[str, float]:
        """Generate one completion and return ``(decoded_text, latency_seconds)``."""
        if self._decoder is not None:
            result = self._decoder.generate(self._encoder.prompt_ids(example), self._max_new_tokens)
            text = cast(
                str, self._loaded.tokenizer.decode(list(result.tokens), skip_special_tokens=True)
            )
            return text, result.latency_s
        tokenizer = self._loaded.tokenizer
        inputs = tokenizer(self._encoder.prompt_text(example), return_tensors="pt").to(
            self._loaded.device
        )
        pad_id = tokenizer.pad_token_id
        start = time.perf_counter()
        with torch.inference_mode():
            output = self._loaded.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        self._sync()
        latency = time.perf_counter() - start
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = cast(str, tokenizer.decode(new_tokens, skip_special_tokens=True))
        return text, latency

    def _sync(self) -> None:
        """Synchronise CUDA so timings are wall-clock honest."""
        if self._loaded.device.type == "cuda":
            torch.cuda.synchronize()


class MetricsCalculator:
    """Classification and evidence metrics that never collapse labels."""

    @staticmethod
    def classification(records: Sequence[EvalRecord]) -> dict[str, Any]:
        """Return accuracy, macro-F1, per-class scores and confusion matrices.

        ``overall`` uses the six distinct labels across both criterion types; ``inclusion`` and
        ``exclusion`` use their own four-label sets.
        """
        overall_labels = list(dict.fromkeys([*c.INCLUSION_LABELS, *c.EXCLUSION_LABELS]))
        blocks: dict[str, Any] = {
            "overall": MetricsCalculator._block(records, overall_labels),
            "inclusion": MetricsCalculator._block(
                [r for r in records if r.criterion_type == c.CRITERION_TYPE_INCLUSION],
                list(c.INCLUSION_LABELS),
            ),
            "exclusion": MetricsCalculator._block(
                [r for r in records if r.criterion_type == c.CRITERION_TYPE_EXCLUSION],
                list(c.EXCLUSION_LABELS),
            ),
        }
        return blocks

    @staticmethod
    def _block(records: Sequence[EvalRecord], labels: list[str]) -> dict[str, Any]:
        """Compute one classification block."""
        if not records:
            return {"n": 0, "labels": labels, "accuracy": 0.0, "report": {}, "confusion": []}
        y_true = [r.gold_label for r in records]
        y_pred = [r.pred_label for r in records]
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        accuracy = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(records)
        return {
            "n": len(records),
            "labels": labels,
            "accuracy": accuracy,
            "report": report,
            "confusion": matrix.tolist(),
        }

    @staticmethod
    def evidence(records: Sequence[EvalRecord]) -> dict[str, Any]:
        """Return micro/macro precision, recall and F1 for evidence sentence sets.

        Empty-set rule: both empty -> P=R=F1=1; exactly one empty -> all three 0.
        """
        return {
            "all": MetricsCalculator._evidence_block(records),
            "non_empty_gold": MetricsCalculator._evidence_block(
                [r for r in records if r.gold_evidence]
            ),
        }

    @staticmethod
    def _evidence_block(records: Sequence[EvalRecord]) -> dict[str, Any]:
        """Aggregate per-record evidence scores, micro (pooled) and macro (mean)."""
        tp = fp = fn = 0
        per_record: list[tuple[float, float, float]] = []
        for record in records:
            gold = set(record.gold_evidence)
            pred = set(record.pred_evidence)
            if not gold and not pred:
                per_record.append((1.0, 1.0, 1.0))
                continue
            if not gold:
                fp += len(pred)
                per_record.append((0.0, 0.0, 0.0))
                continue
            if not pred:
                fn += len(gold)
                per_record.append((0.0, 0.0, 0.0))
                continue
            hit = len(gold & pred)
            tp += hit
            fp += len(pred) - hit
            fn += len(gold) - hit
            precision = hit / len(pred)
            recall = hit / len(gold)
            per_record.append((precision, recall, 2 * precision * recall / (precision + recall)))
        if tp + fp + fn == 0:
            micro = (1.0, 1.0, 1.0)
        else:
            precision_m = tp / (tp + fp) if tp + fp else 0.0
            recall_m = tp / (tp + fn) if tp + fn else 0.0
            f1_m = (
                2 * precision_m * recall_m / (precision_m + recall_m)
                if precision_m + recall_m
                else 0.0
            )
            micro = (precision_m, recall_m, f1_m)
        macro = tuple(
            float(np.mean([scores[i] for scores in per_record])) if per_record else 0.0
            for i in range(3)
        )
        return {
            "n": len(records),
            "n_empty_gold": sum(1 for r in records if not r.gold_evidence),
            "n_empty_pred": sum(1 for r in records if not r.pred_evidence),
            "micro": {"precision": micro[0], "recall": micro[1], "f1": micro[2]},
            "macro": {"precision": macro[0], "recall": macro[1], "f1": macro[2]},
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }


class ConfusionPlotter:
    """Draw confusion-matrix heat maps for the report."""

    @staticmethod
    def plot(block: dict[str, Any], title: str, path: Path) -> None:
        """Write one confusion matrix as a PDF."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matrix = np.asarray(block["confusion"], dtype=float)
        labels = block["labels"]
        figure, axis = plt.subplots(figsize=(5.2, 4.4))
        axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("predicted")
        axis.set_ylabel("gold")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
        axis.set_title(title)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)


class ErrorAnalyzer:
    """Write a stratified markdown file of misclassified examples with cited sentences."""

    def __init__(self, splitter: SentenceSplitter | None = None) -> None:
        """Allow injecting the sentence splitter used to quote evidence."""
        self._splitter = splitter or SentenceSplitter()

    def categorise(self, record: EvalRecord) -> str | None:
        """Return the first matching error category, or ``None`` for a clean prediction."""
        gold, pred = set(record.gold_evidence), set(record.pred_evidence)
        pair = {record.gold_label, record.pred_label}
        if not record.strict and record.pred_label == c.LABEL_INVALID:
            return "invalid_output"
        if pair == {"not applicable", "not enough information"}:
            return "not_applicable_vs_not_enough_information"
        if record.pred_label != record.gold_label:
            return "wrong_label"
        if gold and not pred:
            return "missing_evidence"
        if pred and not gold:
            return "unsupported_evidence"
        if pred - gold:
            return "extra_evidence"
        return None

    def run(self, records: Sequence[EvalRecord], path: Path, per_category: int = 4) -> int:
        """Write at most ``per_category`` cases per category and return how many were written."""
        buckets: dict[str, list[EvalRecord]] = {}
        for record in records:
            category = self.categorise(record)
            if category is not None:
                buckets.setdefault(category, []).append(record)
        lines = ["# Error analysis", "", "Stratified misclassified examples (automatic dump).", ""]
        written = 0
        for category, items in sorted(buckets.items()):
            lines += [f"## {category} ({len(items)} cases)", ""]
            for record in items[:per_category]:
                sentences = self._splitter.split(record.note)
                lines += [
                    f"### annotation {record.annotation_id} · {record.patient_id} · "
                    f"{record.criterion_type}",
                    f"- criterion: {record.criterion_text}",
                    f"- gold: **{record.gold_label}** {list(record.gold_evidence)} · "
                    f"pred: **{record.pred_label}** {list(record.pred_evidence)}",
                    f"- raw output: `{record.raw.strip()[:300]}`",
                    "- gold sentences:",
                    *(
                        f"  - [{i}] {sentences[i]}"
                        for i in record.gold_evidence
                        if i < len(sentences)
                    ),
                    "- predicted sentences:",
                    *(
                        f"  - [{i}] {sentences[i]}"
                        for i in record.pred_evidence
                        if i < len(sentences)
                    ),
                    "",
                ]
                written += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return written


class Evaluator:
    """Run one (split, model tag) evaluation and persist metrics, predictions and figures."""

    def __init__(
        self,
        split: str,
        model_tag: str,
        adapter: str | Path | None = None,
        final: bool = False,
        max_new_tokens: int = c.GENERATION.max_new_tokens,
        limit: int | None = None,
        decode: str = "hf",
        model_name: str = c.BASE_MODEL_NAME,
        file_tag: str | None = None,
    ) -> None:
        """Configure the run; the test split requires ``final=True``."""
        SplitManager.assert_split_allowed(split, final)
        if decode not in c.DECODE_MODES:
            raise ValueError(f"unknown decode mode {decode!r}; choose from {c.DECODE_MODES}")
        self._split = split
        self._model_tag = model_tag
        self._adapter = adapter
        self._max_new_tokens = max_new_tokens
        self._limit = limit
        self._decode = decode
        self._file_tag = file_tag or decode
        self._model_name = model_name

    def run(self) -> dict[str, Any]:
        """Evaluate, write ``results/{split}_{model_tag}_{decode}.json`` and return its content."""
        annotations = AnnotationLoader().load()
        manager = SplitManager()
        manifest = manager.load() if c.SPLIT_MANIFEST_PATH.exists() else manager.build(annotations)
        items = manager.assign(annotations, manifest)[self._split]
        if self._limit is not None:
            items = items[: self._limit]
        loaded = ModelLoader(self._model_name, adapter=self._adapter).load()
        runner = GenerationRunner(loaded, self._max_new_tokens, self._decode)
        builder = PromptBuilder()
        parser = OutputParser()
        splitter = SentenceSplitter()
        records: list[EvalRecord] = []
        for annotation in items:
            raw, latency = runner.run(builder.build(annotation))
            prediction = parser.parse(
                raw, annotation.criterion_type, len(splitter.split(annotation.note))
            )
            records.append(self._record(annotation, prediction, raw, latency))
        metrics = self._metrics(records, loaded)
        c.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{self._split}_{self._model_tag}_{self._file_tag}"
        stem_path = c.RESULTS_DIR / f"{stem}_predictions.jsonl"
        self._write_predictions(records, stem_path)
        self._write_figures(metrics["classification"], stem)
        path = c.RESULTS_DIR / f"{stem}.json"
        path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        return metrics

    @staticmethod
    def _record(
        annotation: Annotation, prediction: Prediction, raw: str, latency: float
    ) -> EvalRecord:
        """Bundle one annotation and its parsed prediction."""
        return EvalRecord(
            annotation_id=annotation.annotation_id,
            patient_id=annotation.patient_id,
            trial_id=annotation.trial_id,
            criterion_type=annotation.criterion_type,
            criterion_text=annotation.criterion_text,
            note=annotation.note,
            gold_label=annotation.label,
            pred_label=prediction.label,
            gold_evidence=annotation.evidence_ids,
            pred_evidence=prediction.evidence_ids,
            raw=raw,
            strict=prediction.strict,
            truncated_ids=prediction.dropped_ids,
            latency_s=latency,
        )

    def _metrics(self, records: Sequence[EvalRecord], loaded: LoadedModel) -> dict[str, Any]:
        """Assemble the full metrics document."""
        n = len(records)
        return {
            "split": self._split,
            "model": self._model_tag,
            "decode": self._decode,
            "n_records": n,
            "seed": c.SEED,
            "base_model": loaded.base_name,
            "adapter": loaded.adapter,
            "max_new_tokens": self._max_new_tokens,
            "classification": MetricsCalculator.classification(records),
            "evidence": MetricsCalculator.evidence(records),
            "parse": {
                "strict_rate": sum(r.strict for r in records) / n if n else 0.0,
                "invalid_rate": sum(r.pred_label == c.LABEL_INVALID for r in records) / n
                if n
                else 0.0,
                "recovered_rate": sum(
                    not r.strict and r.pred_label != c.LABEL_INVALID for r in records
                )
                / n
                if n
                else 0.0,
                "dropped_ids_total": sum(len(r.truncated_ids) for r in records),
            },
            "latency": {
                "mean_s": float(np.mean([r.latency_s for r in records])) if n else 0.0,
                "total_s": float(np.sum([r.latency_s for r in records])),
            },
        }

    @staticmethod
    def _write_predictions(records: Sequence[EvalRecord], path: Path) -> None:
        """Write one JSON object per record."""
        path.write_text(
            "".join(json.dumps(record.to_json()) + "\n" for record in records), encoding="utf-8"
        )

    def _write_figures(self, blocks: dict[str, Any], stem: str) -> None:
        """Write confusion matrices for the overall, inclusion and exclusion blocks."""
        for name, block in blocks.items():
            if block["n"]:
                ConfusionPlotter.plot(
                    block, f"{stem} · {name}", c.FIGURES_DIR / f"{stem}_confusion_{name}.pdf"
                )


def main() -> None:
    """CLI: ``python -m trialmatch.evaluate --split val --model base [--final] [--errors]``."""
    parser = argparse.ArgumentParser(description="Evaluate base or fine-tuned model.")
    parser.add_argument("--split", choices=list(c.SPLIT_NAMES), default="val")
    parser.add_argument(
        "--model", choices=[c.MODEL_TAG_BASE, c.MODEL_TAG_FINETUNED], default="base"
    )
    parser.add_argument(
        "--adapter", type=Path, default=None, help="adapter dir for --model finetuned"
    )
    parser.add_argument("--final", action="store_true", help="required for --split test")
    parser.add_argument("--errors", action="store_true", help="write results/error_cases.md")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    parser.add_argument("--max-new-tokens", type=int, default=c.GENERATION.max_new_tokens)
    parser.add_argument(
        "--smoke", action="store_true", help="0.5B model, 4 records (local wiring check)"
    )
    parser.add_argument(
        "--decode",
        choices=list(c.DECODE_MODES),
        default="hf",
        help="hf = model.generate baseline; cached/uncached = hand-written loops",
    )
    args = parser.parse_args()

    adapter = args.adapter
    if args.model == c.MODEL_TAG_FINETUNED and adapter is None:
        adapter = c.ADAPTER_DIR if c.ADAPTER_DIR.exists() else c.ADAPTER_REPO
    evaluator = Evaluator(
        split=args.split,
        model_tag=args.model,
        adapter=adapter,
        final=args.final,
        max_new_tokens=args.max_new_tokens,
        limit=4 if args.smoke else args.limit,
        model_name=c.SMOKE_MODEL_NAME if args.smoke else c.BASE_MODEL_NAME,
        decode=args.decode,
        file_tag=f"smoke-{args.decode}" if args.smoke else None,
    )
    metrics = evaluator.run()
    print(json.dumps(metrics["classification"]["overall"]["report"].get("macro avg", {}), indent=2))
    if args.errors:
        tag = f"smoke-{args.decode}" if args.smoke else args.decode
        predictions = c.RESULTS_DIR / f"{args.split}_{args.model}_{tag}_predictions.jsonl"
        rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
        written = ErrorAnalyzer().run(
            [EvalRecord.from_json(row) for row in rows], c.RESULTS_DIR / "error_cases.md"
        )
        print(f"wrote {written} error cases")


if __name__ == "__main__":
    main()
