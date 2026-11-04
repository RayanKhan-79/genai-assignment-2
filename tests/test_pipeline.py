"""Tests for the patient-level aggregation rule."""

from __future__ import annotations

import pytest

from trialmatch import constants as c
from trialmatch.pipeline import CriterionLabel, OverallAggregator

INC = c.CRITERION_TYPE_INCLUSION
EXC = c.CRITERION_TYPE_EXCLUSION


def labels(*pairs: tuple[str, str]) -> list[CriterionLabel]:
    """Build criterion labels from ``(type, label)`` pairs."""
    return [CriterionLabel(criterion_type=kind, label=label) for kind, label in pairs]


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        ([(INC, "not included")], c.OUTCOME_INELIGIBLE),
        ([(EXC, "excluded")], c.OUTCOME_INELIGIBLE),
        ([(EXC, "excluded"), (INC, "included")], c.OUTCOME_INELIGIBLE),
        ([(INC, "included"), (EXC, "not excluded")], c.OUTCOME_ELIGIBLE),
        ([(INC, "included")], c.OUTCOME_ELIGIBLE),
        ([(INC, "not enough information")], c.OUTCOME_NEEDS_INFO),
        ([(INC, "included"), (EXC, "not enough information")], c.OUTCOME_NEEDS_INFO),
        ([(INC, "included"), (EXC, c.LABEL_INVALID)], c.OUTCOME_NEEDS_INFO),
    ],
)
def test_aggregation_rule(pairs: list[tuple[str, str]], expected: str) -> None:
    """The documented rule maps per-criterion labels to one outcome."""
    assert OverallAggregator.aggregate(labels(*pairs)) == expected


def test_not_applicable_is_ignored() -> None:
    """`not applicable` criteria never decide the outcome."""
    assert (
        OverallAggregator.aggregate(
            labels((INC, "included"), (EXC, "not applicable"), (EXC, "not excluded"))
        )
        == c.OUTCOME_ELIGIBLE
    )
    assert OverallAggregator.aggregate(labels((EXC, "not applicable"))) == c.OUTCOME_NEEDS_INFO
