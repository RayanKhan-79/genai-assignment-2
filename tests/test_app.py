"""Application helpers: criterion parsing and readable failure messages."""

from __future__ import annotations

import pytest

from trialmatch import constants as c
from trialmatch.app import ScreeningApp, _failure


def test_criteria_split_by_type() -> None:
    """Each non-empty line becomes a criterion of the box's type."""
    specs = ScreeningApp._criteria("age 18+\n\n diabetes ", "pregnant")
    assert [(spec.criterion_type, spec.text) for spec in specs] == [
        (c.CRITERION_TYPE_INCLUSION, "age 18+"),
        (c.CRITERION_TYPE_INCLUSION, "diabetes"),
        (c.CRITERION_TYPE_EXCLUSION, "pregnant"),
    ]


def test_empty_criteria_is_rejected() -> None:
    """Screening without a criterion is a user error with a clear message."""
    with pytest.raises(ValueError, match="at least one criterion"):
        ScreeningApp._criteria("  ", "\n")


def test_too_many_criteria_is_rejected() -> None:
    """The per-run cap is enforced before any model work."""
    with pytest.raises(ValueError, match="too many criteria"):
        ScreeningApp._criteria("\n".join(["x"] * (c.APP_MAX_CRITERIA + 1)), "")


def test_oom_failure_adds_cpu_hint() -> None:
    """An out-of-memory error points the user at the CPU fallback."""
    message = _failure("Could not screen", RuntimeError("Insufficient Memory"))
    assert message.startswith("Could not screen: Insufficient Memory")
    assert "--device cpu" in message


def test_other_failure_is_unchanged() -> None:
    """Ordinary errors are passed through without the GPU hint."""
    assert _failure("Could not screen", ValueError("empty note")) == "Could not screen: empty note"
