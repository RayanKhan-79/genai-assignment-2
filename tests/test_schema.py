"""Tests for the output parser and its tolerant recovery behaviour."""

from __future__ import annotations

import pytest

from trialmatch import constants as c
from trialmatch.schema import OutputParser, Prediction

PARSER = OutputParser()
N_SENTENCES = 5


def parse(raw: str, criterion_type: str = c.CRITERION_TYPE_INCLUSION) -> Prediction:
    """Parse a completion as an inclusion criterion of a five-sentence note."""
    return PARSER.parse(raw, criterion_type, N_SENTENCES)


def test_strict_json() -> None:
    """Plain JSON is parsed without recovery and IDs are sorted."""
    prediction = parse('{"label": "included", "evidence_ids": [3, 1]}')
    assert prediction.valid
    assert prediction.strict
    assert prediction.label == "included"
    assert prediction.evidence_ids == (1, 3)


def test_json_inside_code_fence_is_recovered() -> None:
    """Markdown fences and chatter are tolerated but marked as non-strict."""
    prediction = parse(
        'Sure!\n```json\n{"label": "Not Enough Information", "evidence_ids": []}\n```'
    )
    assert prediction.valid
    assert not prediction.strict
    assert prediction.label == "not enough information"
    assert prediction.evidence_ids == ()


def test_regex_recovery_when_json_is_broken() -> None:
    """A truncated object still yields its label and available IDs."""
    prediction = parse('{"label": "included", "evidence_ids": [0, 2')
    assert prediction.valid
    assert not prediction.strict
    assert prediction.evidence_ids == (0, 2)


def test_garbage_is_invalid() -> None:
    """Unparseable text becomes the invalid label and is a wrong answer."""
    prediction = parse("I think the patient is probably eligible.")
    assert prediction.label == c.LABEL_INVALID
    assert not prediction.valid
    assert not prediction.strict


def test_label_must_belong_to_criterion_type() -> None:
    """An inclusion criterion cannot be answered with an exclusion label."""
    prediction = parse('{"label": "excluded", "evidence_ids": [1]}')
    assert prediction.label == c.LABEL_INVALID
    assert prediction.evidence_ids == (1,)
    assert parse('{"label": "excluded", "evidence_ids": [1]}', c.CRITERION_TYPE_EXCLUSION).valid


def test_out_of_range_and_duplicate_evidence_ids() -> None:
    """IDs outside the note are dropped and reported; duplicates collapse."""
    prediction = parse('{"label": "included", "evidence_ids": [0, 0, 4, 7, -1, "2"]}')
    assert prediction.evidence_ids == (0, 2, 4)
    assert prediction.dropped_ids == (-1, 7)


def test_scalar_evidence_id_is_accepted() -> None:
    """A bare integer instead of a list is normalised."""
    assert parse('{"label": "included", "evidence_ids": 2}').evidence_ids == (2,)


def test_unknown_criterion_type_raises() -> None:
    """Unknown criterion types are a programming error, not a prediction."""
    with pytest.raises(ValueError):
        PARSER.parse("{}", "screening", N_SENTENCES)
