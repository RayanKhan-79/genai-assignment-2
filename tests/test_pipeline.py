"""Tests for the screening result assembly, report export and aggregation rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialmatch import constants as c
from trialmatch.pipeline import (
    CriterionLabel,
    CriterionSpec,
    OverallAggregator,
    Provenance,
    ReportExporter,
    ScreeningPipeline,
)

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


NOTE = "0. The patient has diabetes.\n1. She takes metformin.\n2. She is 60 years old."


def provenance() -> Provenance:
    """Return a fixed provenance block for deterministic report tests."""
    return Provenance(
        base_model="tiny",
        adapter=None,
        decode="cached",
        max_new_tokens=16,
        created_at="2026-11-05T00:00:00+00:00",
    )


def test_invalid_criterion_type_and_empty_text() -> None:
    """Criterion specs are validated before any generation."""
    with pytest.raises(ValueError, match="criterion type"):
        CriterionSpec("outcome", "anything")
    with pytest.raises(ValueError, match="empty"):
        CriterionSpec(INC, "   ")


def test_assemble_maps_labels_evidence_and_outcome() -> None:
    """Assembly parses completions, resolves evidence text and applies the rule."""
    criteria = [
        CriterionSpec(INC, "Has diabetes"),
        CriterionSpec(EXC, "Is pregnant"),
    ]
    completions = [
        ('{"label": "included", "evidence_ids": [0, 1]}', 0.5),
        ('{"label": "not excluded", "evidence_ids": []}', 0.25),
    ]
    result = ScreeningPipeline.assemble(NOTE, criteria, completions, provenance())
    assert result.outcome == c.OUTCOME_ELIGIBLE
    assert result.items[0].evidence_ids == (0, 1)
    assert result.items[0].evidence_text == (
        "The patient has diabetes.",
        "She takes metformin.",
    )
    assert result.items[0].explanation == c.APP_EXPLANATIONS["included"]
    assert result.items[0].strict is True
    assert sum(item.latency_s for item in result.items) == pytest.approx(0.75)
    assert result.missing_information == ()


def test_assemble_collects_missing_information_and_invalid_output() -> None:
    """Uncertain and unparseable predictions appear in the missing-information list."""
    criteria = [CriterionSpec(INC, "Has diabetes"), CriterionSpec(EXC, "Is pregnant")]
    completions = [
        ('{"label": "not enough information", "evidence_ids": []}', 0.1),
        ("I cannot answer that.", 0.1),
    ]
    result = ScreeningPipeline.assemble(NOTE, criteria, completions, provenance())
    assert result.outcome == c.OUTCOME_NEEDS_INFO
    assert result.items[1].label == c.LABEL_INVALID
    assert len(result.missing_information) == 2
    assert any(INC in entry for entry in result.missing_information)
    assert any(EXC in entry for entry in result.missing_information)


def test_assemble_requires_one_completion_per_criterion() -> None:
    """A mismatched number of completions is a programming error, not a silent skip."""
    with pytest.raises(ValueError, match="one completion"):
        ScreeningPipeline.assemble(NOTE, [CriterionSpec(INC, "Has diabetes")], [], provenance())


def test_report_contains_disclaimer_rule_and_evidence(tmp_path: Path) -> None:
    """Markdown and HTML reports carry the disclaimer, rule, evidence and provenance."""
    criteria = [CriterionSpec(INC, "Has diabetes")]
    completions = [('{"label": "included", "evidence_ids": [0]}', 0.5)]
    result = ScreeningPipeline.assemble(NOTE, criteria, completions, provenance())
    markdown = ReportExporter.to_markdown(result)
    page = ReportExporter.to_html(result)
    assert c.APP_DISCLAIMER in markdown
    assert c.APP_AGGREGATION_RULE in markdown
    assert "The patient has diabetes." in markdown
    assert "included" in markdown
    assert "<strong>" in page and c.APP_DISCLAIMER in page
    assert "&lt;" not in page  # escaping must not mangle ordinary text
    written_md, written_html = ReportExporter.write(result, tmp_path)
    assert written_md.exists() and written_html.exists()
    assert written_md.read_text(encoding="utf-8") == markdown


def test_report_escapes_html_and_table_pipes() -> None:
    """User text cannot break the HTML page or the Markdown table."""
    criteria = [CriterionSpec(INC, "Has <script>alert(1)</script> | diabetes")]
    completions = [('{"label": "included", "evidence_ids": [0]}', 0.1)]
    result = ScreeningPipeline.assemble(NOTE, criteria, completions, provenance())
    assert "<script>" not in ReportExporter.to_html(result)
    assert "&lt;script&gt;" in ReportExporter.to_html(result)
    assert "\\|" in ReportExporter.to_markdown(result)
