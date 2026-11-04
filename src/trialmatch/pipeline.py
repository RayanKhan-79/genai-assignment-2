"""Patient-level screening pipeline and report export.

Phase 3 only needs the documented aggregation rule; the full screening pipeline and the report
exporter are added in Phase 7.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from trialmatch import constants as c


@dataclass(frozen=True)
class CriterionLabel:
    """One criterion's type and predicted label."""

    criterion_type: str
    label: str


class OverallAggregator:
    """Turn per-criterion labels into a preliminary patient-level outcome.

    The rule is an explicit, documented heuristic — never a ground-truth label:

    1. ``not applicable`` criteria are ignored.
    2. Any inclusion ``not included`` or exclusion ``excluded`` -> ``Likely ineligible``.
    3. Any ``not enough information`` (or unusable label) -> ``Needs more information``.
    4. All inclusions ``included`` and all exclusions ``not excluded`` -> ``Likely eligible``.
    5. Otherwise (including "no applicable criteria") -> ``Needs more information``.
    """

    @staticmethod
    def aggregate(labels: Sequence[CriterionLabel]) -> str:
        """Apply the rule and return one of the three ``OUTCOME_*`` constants."""
        active = [item for item in labels if item.label != "not applicable"]
        if not active:
            return c.OUTCOME_NEEDS_INFO
        for item in active:
            contradicts = (
                item.criterion_type == c.CRITERION_TYPE_INCLUSION and item.label == "not included"
            ) or (item.criterion_type == c.CRITERION_TYPE_EXCLUSION and item.label == "excluded")
            if contradicts:
                return c.OUTCOME_INELIGIBLE
        for item in active:
            uncertain = item.label in {"not enough information", c.LABEL_INVALID}
            if uncertain:
                return c.OUTCOME_NEEDS_INFO
        all_included = all(
            item.label == "included"
            for item in active
            if item.criterion_type == c.CRITERION_TYPE_INCLUSION
        )
        all_excluded_not_met = all(
            item.label == "not excluded"
            for item in active
            if item.criterion_type == c.CRITERION_TYPE_EXCLUSION
        )
        if all_included and all_excluded_not_met:
            return c.OUTCOME_ELIGIBLE
        return c.OUTCOME_NEEDS_INFO
