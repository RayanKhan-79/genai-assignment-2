"""Sanity checks on the shared constants (rule R4: one source of truth)."""

from __future__ import annotations

from trialmatch import constants as c


def test_label_sets_are_disjoint_and_complete() -> None:
    """Every allowed label belongs to exactly one criterion type."""
    inclusion = set(c.INCLUSION_LABELS)
    exclusion = set(c.EXCLUSION_LABELS)
    assert len(inclusion) == len(c.INCLUSION_LABELS)
    assert len(exclusion) == len(c.EXCLUSION_LABELS)
    assert {"not enough information", "not applicable"} <= inclusion & exclusion
    assert "included" in inclusion and "included" not in exclusion
    assert "excluded" in exclusion and "excluded" not in inclusion
    assert c.LABEL_INVALID not in inclusion | exclusion


def test_split_sizes_match_patient_total() -> None:
    """The fixed 37/8/8 split must cover the 53 annotated patients."""
    assert c.N_TRAIN_PATIENTS + c.N_VAL_PATIENTS + c.N_TEST_PATIENTS == 53
    assert c.SPLIT_NAMES == ("train", "val", "test")


def test_paths_are_absolute_and_inside_project() -> None:
    """All artefact paths resolve under the project root."""
    for path in (c.RESULTS_DIR, c.FIGURES_DIR, c.SPLITS_DIR, c.ADAPTER_DIR):
        assert path.is_absolute()
        assert c.PROJECT_ROOT in path.parents
