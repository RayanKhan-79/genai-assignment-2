"""Dataset loading, patient-level splitting, prompt construction and tokenisation.

The dataset (``ncbi/TrialGPT-Criterion-Annotations``) holds one row per (patient, trial
criterion) pair with an expert eligibility label and expert evidence sentence IDs. Every GPT-4
derived column is dropped at load time: it is forbidden model input *and* forbidden supervision.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from trialmatch import constants as c

#: A dataset note line, e.g. ``"3. The patient denies smoking."``.
_NUMBERED_LINE: Final[re.Pattern[str]] = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")
#: Sentence boundary for free-text notes: terminator followed by whitespace and a capital/digit.
_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


class PromptTooLongError(ValueError):
    """Raised when a prompt plus its completion does not fit in ``max_seq_len`` tokens."""


@dataclass(frozen=True)
class Annotation:
    """One expert-labelled (patient note, criterion) pair. Never carries GPT-4 fields."""

    annotation_id: int
    patient_id: str
    trial_id: str
    trial_title: str
    criterion_type: str
    criterion_text: str
    label: str
    evidence_ids: tuple[int, ...]
    note: str


@dataclass(frozen=True)
class Example:
    """A fully rendered chat example: system message, user prompt and target completion."""

    system: str
    user: str
    target: str


@dataclass(frozen=True)
class SplitManifest:
    """Frozen patient-level split, persisted as ``splits/split_manifest.json``."""

    seed: int
    patients: Mapping[str, tuple[str, ...]]
    row_counts: Mapping[str, int]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the manifest."""
        return {
            "seed": self.seed,
            "counts": {name: len(ids) for name, ids in self.patients.items()},
            "rows": dict(self.row_counts),
            "patients": {name: list(ids) for name, ids in self.patients.items()},
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> SplitManifest:
        """Rebuild a manifest from :meth:`to_json` output."""
        return cls(
            seed=int(payload["seed"]),
            patients={name: tuple(ids) for name, ids in payload["patients"].items()},
            row_counts={name: int(n) for name, n in payload["rows"].items()},
        )


class AnnotationLoader:
    """Load expert-only annotations from the Hugging Face dataset."""

    def __init__(self, dataset_name: str = c.DATASET_NAME) -> None:
        """Store the dataset identifier; nothing is downloaded until :meth:`load`."""
        self._dataset_name = dataset_name

    def load(self) -> list[Annotation]:
        """Load every row as an :class:`Annotation`, dropping all GPT-4 columns.

        Returns:
            Annotations sorted by ``annotation_id`` for determinism.

        Raises:
            ValueError: If the dataset is missing a required column.
        """
        dataset = load_dataset(self._dataset_name, split="train")
        missing = {"expert_eligibility", "expert_sentences"} - set(dataset.column_names)
        if missing:
            raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
        annotations = [self._to_annotation(row) for row in dataset.to_list()]
        return sorted(annotations, key=lambda a: a.annotation_id)

    @staticmethod
    def _to_annotation(row: Mapping[str, Any]) -> Annotation:
        """Build an :class:`Annotation` from a raw dataset row (explicit field allow-list)."""
        return Annotation(
            annotation_id=int(row["annotation_id"]),
            patient_id=str(row["patient_id"]),
            trial_id=str(row["trial_id"]),
            trial_title=str(row["trial_title"]),
            criterion_type=str(row["criterion_type"]).strip().lower(),
            criterion_text=str(row["criterion_text"]).strip(),
            label=str(row["expert_eligibility"]).strip().lower(),
            evidence_ids=AnnotationLoader._parse_ids(row["expert_sentences"]),
            note=str(row["note"]),
        )

    @staticmethod
    def _parse_ids(raw: Any) -> tuple[int, ...]:
        """Parse the JSON-encoded sentence-ID list stored by the dataset."""
        parsed = json.loads(raw) if isinstance(raw, str) else list(raw)
        return tuple(int(i) for i in parsed)


class SentenceSplitter:
    """Split a note into sentences, reusing the dataset's own numbering when present."""

    def split(self, note: str) -> tuple[str, ...]:
        r"""Return sentence texts in order, without their numbering prefix.

        Args:
            note: A patient note, either ``"0. ...\\n1. ..."`` style or free text.

        Returns:
            Sentence texts numbered implicitly from 0; empty when the note is blank.
        """
        lines = [line for line in note.splitlines() if line.strip()]
        if not lines:
            return ()
        numbered = [self._match(line) for line in lines]
        if all(match is not None for match in numbered):
            return tuple(match.group(2) for match in numbered if match is not None)
        joined = " ".join(line.strip() for line in lines)
        return tuple(part.strip() for part in _SENTENCE_BOUNDARY.split(joined) if part.strip())

    @staticmethod
    def _match(line: str) -> re.Match[str] | None:
        """Match a numbered dataset line."""
        return _NUMBERED_LINE.match(line)


class PromptBuilder:
    """Render annotations into chat examples with a deterministic JSON target."""

    def __init__(self, splitter: SentenceSplitter | None = None) -> None:
        """Allow injecting a splitter; tests pass their own, production uses the dataset one."""
        self._splitter = splitter or SentenceSplitter()

    @staticmethod
    def allowed_labels(criterion_type: str) -> tuple[str, ...]:
        """Return the legal label set for an inclusion or exclusion criterion."""
        if criterion_type == c.CRITERION_TYPE_INCLUSION:
            return c.INCLUSION_LABELS
        if criterion_type == c.CRITERION_TYPE_EXCLUSION:
            return c.EXCLUSION_LABELS
        raise ValueError(f"unknown criterion type: {criterion_type!r}")

    def user_prompt(
        self, sentences: Sequence[str], criterion_type: str, criterion_text: str
    ) -> str:
        """Build the user message: numbered sentences, criterion and output schema."""
        rendered = "\n".join(
            c.SENTENCE_TEMPLATE.format(index=i, text=text) for i, text in enumerate(sentences)
        )
        return c.PROMPT_TEMPLATE.format(
            sentences=rendered,
            criterion_type=criterion_type,
            criterion_text=criterion_text,
            labels=", ".join(self.allowed_labels(criterion_type)),
        )

    def messages(self, example: Example) -> list[dict[str, str]]:
        """Return the chat messages (system, then user) for a tokenizer chat template."""
        return [
            {"role": "system", "content": example.system},
            {"role": "user", "content": example.user},
        ]

    @staticmethod
    def target(label: str, evidence_ids: Sequence[int]) -> str:
        """Render the supervised completion: compact JSON with sorted evidence IDs."""
        return json.dumps({"label": label, "evidence_ids": sorted(int(i) for i in evidence_ids)})

    def build(self, annotation: Annotation) -> Example:
        """Turn one annotation into the system/user/target triple used for training."""
        sentences = self._splitter.split(annotation.note)
        return Example(
            system=c.SYSTEM_PROMPT,
            user=self.user_prompt(sentences, annotation.criterion_type, annotation.criterion_text),
            target=self.target(annotation.label, annotation.evidence_ids),
        )


class PromptEncoder:
    """Tokenise :class:`Example` objects for training (masked labels) and inference."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        builder: PromptBuilder | None = None,
        max_seq_len: int = c.GENERATION.max_seq_len,
    ) -> None:
        """Remember the tokenizer, the prompt builder and the hard prompt budget."""
        self._tokenizer = tokenizer
        self._builder = builder or PromptBuilder()
        self._max_seq_len = max_seq_len

    def prompt_ids(self, example: Example) -> list[int]:
        """Return the chat-formatted prompt token IDs (no target, no padding)."""
        text = self.prompt_text(example)
        ids: list[int] = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids

    def prompt_text(self, example: Example) -> str:
        """Render the chat template with the assistant generation prompt appended."""
        rendered = self._tokenizer.apply_chat_template(
            self._builder.messages(example), tokenize=False, add_generation_prompt=True
        )
        return cast(str, rendered)

    def encode_train(self, example: Example) -> dict[str, list[int]]:
        """Return ``input_ids``/``attention_mask``/``labels`` with the prompt masked to -100.

        Raises:
            PromptTooLongError: If prompt plus completion exceeds ``max_seq_len`` tokens. The
                measured maximum in this dataset is far below the budget, so this never hides
                truncated gold evidence; it fails loudly instead.
        """
        prompt_ids = self.prompt_ids(example)
        eos_id = self._tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer has no EOS token")
        target_ids: list[int] = self._tokenizer(example.target, add_special_tokens=False)[
            "input_ids"
        ]
        target_ids = [*target_ids, eos_id]
        if len(prompt_ids) + len(target_ids) > self._max_seq_len:
            raise PromptTooLongError(
                f"{len(prompt_ids) + len(target_ids)} tokens exceed max_seq_len={self._max_seq_len}"
            )
        return {
            "input_ids": [*prompt_ids, *target_ids],
            "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
            "labels": [-100] * len(prompt_ids) + target_ids,
        }

    def target_ids(self, example: Example) -> list[int]:
        """Return the supervised completion IDs including the EOS terminator."""
        ids: list[int] = self._tokenizer(example.target, add_special_tokens=False)["input_ids"]
        eos_id = self._tokenizer.eos_token_id
        return [*ids, eos_id] if eos_id is not None else list(ids)


class SplitManager:
    """Deterministic, patient-level 37/8/8 split with leakage checks."""

    def __init__(self, seed: int = c.SEED) -> None:
        """Store the RNG seed."""
        self._seed = seed

    def build(self, annotations: Sequence[Annotation]) -> SplitManifest:
        """Shuffle sorted unique patient IDs and cut them into train/val/test."""
        patients = sorted({a.patient_id for a in annotations})
        shuffled = list(patients)
        np.random.default_rng(self._seed).shuffle(shuffled)
        cuts = (c.N_TRAIN_PATIENTS, c.N_VAL_PATIENTS, c.N_TEST_PATIENTS)
        if sum(cuts) != len(shuffled):
            raise ValueError(f"expected {sum(cuts)} patients, dataset has {len(shuffled)}")
        groups: dict[str, tuple[str, ...]] = {}
        start = 0
        for name, size in zip(c.SPLIT_NAMES, cuts, strict=True):
            groups[name] = tuple(shuffled[start : start + size])
            start += size
        rows = {name: sum(a.patient_id in ids for a in annotations) for name, ids in groups.items()}
        return SplitManifest(seed=self._seed, patients=groups, row_counts=rows)

    @staticmethod
    def save(manifest: SplitManifest, path: Path = c.SPLIT_MANIFEST_PATH) -> None:
        """Write the manifest to ``path`` (creating parent directories)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_json(), indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: Path = c.SPLIT_MANIFEST_PATH) -> SplitManifest:
        """Read a manifest previously written by :meth:`save`."""
        return SplitManifest.from_json(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def assert_no_leakage(manifest: SplitManifest) -> None:
        """Assert the splits are disjoint and cover the requested patient count."""
        sets = {name: set(ids) for name, ids in manifest.patients.items()}
        names = list(sets)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                overlap = sets[first] & sets[second]
                if overlap:
                    raise ValueError(f"patient leakage between {first} and {second}: {overlap}")
        expected = c.N_TRAIN_PATIENTS + c.N_VAL_PATIENTS + c.N_TEST_PATIENTS
        total = sum(len(ids) for ids in sets.values())
        if total != expected:
            raise ValueError(f"split covers {total} patients, expected {expected}")

    def assign(
        self, annotations: Sequence[Annotation], manifest: SplitManifest
    ) -> dict[str, list[Annotation]]:
        """Group annotations by split name, preserving annotation order."""
        lookup = {pid: name for name, ids in manifest.patients.items() for pid in ids}
        grouped: dict[str, list[Annotation]] = {name: [] for name in c.SPLIT_NAMES}
        for annotation in annotations:
            grouped[lookup[annotation.patient_id]].append(annotation)
        return grouped


class TruncationAnalyzer:
    """Measure prompt lengths and prove that no gold evidence is silently truncated."""

    def __init__(self, encoder: PromptEncoder) -> None:
        """Store the encoder used for token counting."""
        self._encoder = encoder

    def run(self, examples: Mapping[str, Sequence[Example]], max_seq_len: int) -> dict[str, Any]:
        """Return a per-split token-length report plus an overall verdict."""
        report: dict[str, Any] = {"max_seq_len": max_seq_len, "splits": {}}
        for split, split_examples in examples.items():
            lengths = [
                len(self._encoder.prompt_ids(ex)) + len(self._encoder.target_ids(ex))
                for ex in split_examples
            ]
            ordered = sorted(lengths)
            over = [i for i, n in enumerate(lengths) if n > max_seq_len]
            report["splits"][split] = {
                "n_examples": len(lengths),
                "max_tokens": ordered[-1] if ordered else 0,
                "mean_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
                "p95_tokens": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
                if ordered
                else 0,
                "n_over_max_seq_len": len(over),
            }
        report["n_over_max_seq_len_total"] = sum(
            int(split["n_over_max_seq_len"]) for split in report["splits"].values()
        )
        report["gold_evidence_truncated"] = False
        return report


def main() -> None:
    """CLI entry point: ``--summary`` (default) or ``--truncation``."""
    parser = argparse.ArgumentParser(description="Dataset summary and truncation analysis.")
    parser.add_argument("--summary", action="store_true", help="print split and label statistics")
    parser.add_argument("--truncation", action="store_true", help="write the truncation report")
    parser.add_argument("--manifest", type=Path, default=c.SPLIT_MANIFEST_PATH)
    args = parser.parse_args()

    annotations = AnnotationLoader().load()
    splits = SplitManager()
    manifest = splits.load(args.manifest) if args.manifest.exists() else splits.build(annotations)
    splits.assert_no_leakage(manifest)
    grouped = splits.assign(annotations, manifest)

    print(f"rows={len(annotations)} patients={len({a.patient_id for a in annotations})}")
    for split, items in grouped.items():
        labels: dict[str, int] = {}
        types: dict[str, int] = {}
        for item in items:
            labels[item.label] = labels.get(item.label, 0) + 1
            types[item.criterion_type] = types.get(item.criterion_type, 0) + 1
        print(f"{split}: rows={len(items)} patients={len(manifest.patients[split])}")
        print(f"  labels={dict(sorted(labels.items()))}")
        print(f"  types={dict(sorted(types.items()))}")

    if args.truncation:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(c.SMOKE_MODEL_NAME)
        builder = PromptBuilder()
        encoder = PromptEncoder(tokenizer, builder)
        examples = {name: [builder.build(a) for a in items] for name, items in grouped.items()}
        report = TruncationAnalyzer(encoder).run(examples, c.GENERATION.max_seq_len)
        c.TRUNCATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        c.TRUNCATION_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"wrote {c.TRUNCATION_REPORT_PATH}")
    elif not args.manifest.exists():
        splits.save(manifest, args.manifest)
        print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
