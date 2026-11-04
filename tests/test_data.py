"""Tests for dataset loading, splits, prompt construction and tokenisation."""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any, cast

import pytest
from transformers import PreTrainedTokenizerBase

from trialmatch import constants as c
from trialmatch.data import (
    Annotation,
    AnnotationLoader,
    Example,
    PromptBuilder,
    PromptEncoder,
    PromptTooLongError,
    SentenceSplitter,
    SplitManager,
    SplitManifest,
)

NOTE = (
    "0. A 58-year-old woman presents with chest pain.\n"
    "1. She denies smoking and diabetes.\n"
    "2. She takes no medications."
)
RAW_ROW: dict[str, Any] = {
    "annotation_id": 7,
    "patient_id": "p1",
    "trial_id": "NCT1",
    "trial_title": "A trial",
    "criterion_type": "Inclusion",
    "criterion_text": "Age above 18.",
    "expert_eligibility": "Included",
    "expert_sentences": "[0, 1]",
    "note": NOTE,
    "gpt4_explanation": "LEAK-ME",
    "gpt4_eligibility": "included",
    "gpt4_sentences": "[0]",
    "explanation_correctness": "Correct",
}


def make_annotation(annotation_id: int = 0, patient_id: str = "p1") -> Annotation:
    """Build a small annotation for pure-python tests."""
    return Annotation(
        annotation_id=annotation_id,
        patient_id=patient_id,
        trial_id="NCT1",
        trial_title="A trial",
        criterion_type=c.CRITERION_TYPE_INCLUSION,
        criterion_text="Age above 18.",
        label="included",
        evidence_ids=(0, 2),
        note=NOTE,
    )


def test_loader_drops_gpt4_fields() -> None:
    """No GPT-4 column survives into the Annotation dataclass."""
    annotation = AnnotationLoader._to_annotation(RAW_ROW)
    names = {field.name for field in fields(Annotation)}
    assert not names & set(c.FORBIDDEN_FIELDS)
    assert annotation.label == "included"
    assert annotation.criterion_type == "inclusion"
    assert annotation.evidence_ids == (0, 1)
    assert not any("LEAK-ME" in str(getattr(annotation, name)) for name in names)


def test_prompt_contains_no_leakage() -> None:
    """Rendered prompts and targets never contain GPT-4 text."""
    example = PromptBuilder().build(make_annotation())
    assert "LEAK-ME" not in example.user
    assert "gpt4" not in example.user.lower()
    assert example.target == '{"label": "included", "evidence_ids": [0, 2]}'


def test_sentence_splitter_numbered_and_free_text() -> None:
    """Dataset numbering is stripped; free text falls back to sentence boundaries."""
    splitter = SentenceSplitter()
    assert splitter.split(NOTE) == (
        "A 58-year-old woman presents with chest pain.",
        "She denies smoking and diabetes.",
        "She takes no medications.",
    )
    free = "The patient is 58. She denies smoking! Any allergies?"
    assert splitter.split(free) == (
        "The patient is 58.",
        "She denies smoking!",
        "Any allergies?",
    )
    assert splitter.split("   \n ") == ()


def test_prompt_builder_labels_and_numbering() -> None:
    """Prompts number sentences from 0 and list only the labels of the criterion type."""
    builder = PromptBuilder()
    sentences = SentenceSplitter().split(NOTE)
    inclusion = builder.user_prompt(sentences, c.CRITERION_TYPE_INCLUSION, "Age above 18.")
    exclusion = builder.user_prompt(sentences, c.CRITERION_TYPE_EXCLUSION, "Pregnant.")
    assert "[0] A 58-year-old woman presents with chest pain." in inclusion
    assert "[2] She takes no medications." in inclusion
    assert "included" in inclusion and "excluded" not in inclusion
    assert "excluded" in exclusion and "included" not in exclusion
    with pytest.raises(ValueError):
        builder.allowed_labels("screening")


def test_split_manager_is_deterministic_and_disjoint(tmp_path: Any) -> None:
    """The same seed gives the same 37/8/8 split; overlaps are rejected."""
    annotations = [make_annotation(annotation_id=i, patient_id=f"p{i % 53}") for i in range(120)]
    manager = SplitManager()
    first = manager.build(annotations)
    second = SplitManager(seed=c.SEED).build(annotations)
    assert first.patients == second.patients
    assert [len(first.patients[name]) for name in c.SPLIT_NAMES] == [37, 8, 8]
    manager.assert_no_leakage(first)
    grouped = manager.assign(annotations, first)
    assert sum(len(items) for items in grouped.values()) == len(annotations)

    path = tmp_path / "manifest.json"
    manager.save(first, path)
    assert SplitManager.load(path).patients == first.patients

    broken = SplitManifest(
        seed=first.seed,
        patients={**first.patients, "val": first.patients["val"] + first.patients["train"][:1]},
        row_counts=first.row_counts,
    )
    with pytest.raises(ValueError, match="leakage"):
        manager.assert_no_leakage(broken)


class _CharTokenizer:
    """Minimal stand-in tokenizer: one token per character, EOS = 0."""

    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [1 + ord(ch) for ch in text]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def test_prompt_encoder_masks_prompt_tokens() -> None:
    """Labels are -100 over the prompt and equal the target IDs (plus EOS) afterwards."""
    encoder = PromptEncoder(cast(PreTrainedTokenizerBase, _CharTokenizer()))
    example = Example(system="s", user="u", target="answer")
    encoded = encoder.encode_train(example)
    n_prompt = len(encoder.prompt_ids(example))
    assert encoded["labels"][:n_prompt] == [-100] * n_prompt
    assert encoded["labels"][n_prompt:] == encoder.target_ids(example)
    assert encoded["labels"][-1] == 0
    assert len(encoded["input_ids"]) == len(encoded["labels"])
    assert set(encoded["attention_mask"]) == {1}


def test_prompt_encoder_refuses_oversized_example() -> None:
    """An example above the token budget fails loudly instead of dropping evidence."""
    encoder = PromptEncoder(cast(PreTrainedTokenizerBase, _CharTokenizer()), max_seq_len=10)
    with pytest.raises(PromptTooLongError):
        encoder.encode_train(Example(system="s", user="u", target="answer"))


def test_csv_friendly_json_target() -> None:
    """The target is valid JSON with sorted, de-duplicated-by-construction evidence IDs."""
    target = PromptBuilder.target("not included", [5, 3])
    assert json.loads(target) == {"label": "not included", "evidence_ids": [3, 5]}


def test_split_guard_requires_final_for_test() -> None:
    """The test split is reachable only through --final."""
    SplitManager.assert_split_allowed("val", final=False)
    SplitManager.assert_split_allowed("test", final=True)
    with pytest.raises(PermissionError):
        SplitManager.assert_split_allowed("test", final=False)
    with pytest.raises(ValueError):
        SplitManager.assert_split_allowed("dev", final=False)
