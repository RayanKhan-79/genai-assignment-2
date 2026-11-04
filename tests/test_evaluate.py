"""Tests for the metric calculator (classification shapes and the evidence empty-set rule)."""

from __future__ import annotations

from trialmatch import constants as c
from trialmatch.evaluate import EvalRecord, MetricsCalculator


def record(
    gold_label: str = "included",
    pred_label: str = "included",
    gold_evidence: tuple[int, ...] = (0,),
    pred_evidence: tuple[int, ...] = (0,),
    criterion_type: str = c.CRITERION_TYPE_INCLUSION,
) -> EvalRecord:
    """Build a minimal evaluation record."""
    return EvalRecord(
        annotation_id=0,
        patient_id="p",
        trial_id="NCT1",
        criterion_type=criterion_type,
        criterion_text="x",
        note="0. a\n1. b",
        gold_label=gold_label,
        pred_label=pred_label,
        gold_evidence=gold_evidence,
        pred_evidence=pred_evidence,
        raw="",
        strict=True,
        truncated_ids=(),
        latency_s=0.0,
    )


def test_evidence_empty_set_rule() -> None:
    """Both empty -> 1; exactly one empty -> 0; partial overlap -> harmonic mean."""
    perfect = MetricsCalculator._evidence_block([record(gold_evidence=(), pred_evidence=())])
    assert perfect["macro"]["f1"] == 1.0

    missed = MetricsCalculator._evidence_block([record(gold_evidence=(0,), pred_evidence=())])
    assert missed["macro"]["f1"] == 0.0
    assert missed["micro"]["recall"] == 0.0
    assert missed["fn"] == 1

    spurious = MetricsCalculator._evidence_block([record(gold_evidence=(), pred_evidence=(1,))])
    assert spurious["macro"]["precision"] == 0.0
    assert spurious["fp"] == 1

    partial = MetricsCalculator._evidence_block(
        [record(gold_evidence=(0, 1), pred_evidence=(0, 2))]
    )
    assert partial["micro"]["precision"] == partial["micro"]["recall"] == 0.5
    assert partial["micro"]["f1"] == 0.5


def test_micro_and_macro_agree_when_degenerate() -> None:
    """A fully missed batch must not report a perfect micro score."""
    records = [record(gold_evidence=(0,), pred_evidence=()) for _ in range(3)]
    block = MetricsCalculator._evidence_block(records)
    assert block["micro"]["recall"] == 0.0
    assert block["macro"]["f1"] == 0.0


def test_classification_keeps_four_labels_per_type() -> None:
    """Classification blocks keep all four categories, including absent ones."""
    records = [
        record(gold_label="included", pred_label="included"),
        record(gold_label="not applicable", pred_label="not enough information"),
        record(
            gold_label="not excluded",
            pred_label="not excluded",
            criterion_type=c.CRITERION_TYPE_EXCLUSION,
        ),
    ]
    blocks = MetricsCalculator.classification(records)
    assert set(blocks["inclusion"]["labels"]) == set(c.INCLUSION_LABELS)
    assert set(blocks["exclusion"]["labels"]) == set(c.EXCLUSION_LABELS)
    assert len(blocks["overall"]["labels"]) == 6
    assert blocks["overall"]["n"] == 3
    assert blocks["overall"]["accuracy"] == 2 / 3


def test_invalid_prediction_counts_as_wrong() -> None:
    """An invalid prediction lowers accuracy but never invents a class."""
    block = MetricsCalculator.classification(
        [record(gold_label="included", pred_label=c.LABEL_INVALID)]
    )
    assert block["inclusion"]["accuracy"] == 0.0
    assert block["inclusion"]["report"]["included"]["recall"] == 0.0
