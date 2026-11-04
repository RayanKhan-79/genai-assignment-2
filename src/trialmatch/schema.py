"""Prediction schema and the tolerant parser for model output.

The model is asked for one compact JSON object. :class:`OutputParser` parses it strictly first,
falls back to a regex recovery pass, and records *why* a prediction had to be repaired. Anything
that still yields no legal label becomes :data:`constants.LABEL_INVALID`, which counts as a wrong
label in every metric.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Final

from trialmatch import constants as c

#: Largest ``{...}`` span in a completion, used by the recovery pass.
_OBJECT_SPAN: Final[re.Pattern[str]] = re.compile(r"\{.*\}", re.DOTALL)
_LABEL_FIELD: Final[re.Pattern[str]] = re.compile(r'"label"\s*:\s*"([^"]*)"')
_IDS_FIELD: Final[re.Pattern[str]] = re.compile(r'"evidence[^"]*"\s*:\s*\[([^\]]*)\]?')
_INTEGER: Final[re.Pattern[str]] = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Prediction:
    """A parsed model answer for one criterion.

    Attributes:
        label: Normalised label, or :data:`constants.LABEL_INVALID`.
        evidence_ids: Sorted, de-duplicated sentence IDs that exist in the patient note.
        raw: The raw completion, kept for error analysis.
        strict: Whether the raw completion was directly valid JSON.
        dropped_ids: Evidence IDs removed because they fall outside the note.
    """

    label: str
    evidence_ids: tuple[int, ...]
    raw: str
    strict: bool
    dropped_ids: tuple[int, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether a legal label was recovered."""
        return self.label != c.LABEL_INVALID


class OutputParser:
    """Parse model completions into :class:`Prediction` objects."""

    def parse(self, raw: str, criterion_type: str, n_sentences: int) -> Prediction:
        """Parse one completion.

        Args:
            raw: The completion text produced by the model.
            criterion_type: ``inclusion`` or ``exclusion``; selects the legal label set.
            n_sentences: Number of sentences in the note, used to drop out-of-range evidence IDs.

        Returns:
            A prediction. Unparseable output yields ``label="invalid"`` and ``strict=False``.
        """
        allowed = self.allowed_labels(criterion_type)
        payload, strict = self._payload(raw)
        if payload is None:
            return Prediction(label=c.LABEL_INVALID, evidence_ids=(), raw=raw, strict=False)
        label, ids = payload
        evidence = self._clean_ids(ids, n_sentences)
        if label not in allowed:
            return Prediction(
                label=c.LABEL_INVALID,
                evidence_ids=evidence.kept,
                raw=raw,
                strict=strict,
                dropped_ids=evidence.dropped,
            )
        return Prediction(
            label=label,
            evidence_ids=evidence.kept,
            raw=raw,
            strict=strict,
            dropped_ids=evidence.dropped,
        )

    @staticmethod
    def allowed_labels(criterion_type: str) -> tuple[str, ...]:
        """Return the legal label set for a criterion type (mirrors the prompt)."""
        if criterion_type == c.CRITERION_TYPE_INCLUSION:
            return c.INCLUSION_LABELS
        if criterion_type == c.CRITERION_TYPE_EXCLUSION:
            return c.EXCLUSION_LABELS
        raise ValueError(f"unknown criterion type: {criterion_type!r}")

    def _payload(self, raw: str) -> tuple[tuple[str, list[Any]] | None, bool]:
        """Extract ``(label, evidence_ids)``, preferring a strict JSON parse."""
        text = raw.strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            return self._from_dict(parsed), True
        match = _OBJECT_SPAN.search(text)
        if match is not None:
            try:
                recovered = json.loads(match.group(0))
            except json.JSONDecodeError:
                recovered = None
            if isinstance(recovered, dict):
                return self._from_dict(recovered), False
        label_match = _LABEL_FIELD.search(text)
        if label_match is None:
            return None, False
        ids_match = _IDS_FIELD.search(text)
        ids: list[Any] = [int(n) for n in _INTEGER.findall(ids_match.group(1))] if ids_match else []
        return (self._normalise(label_match.group(1)), ids), False

    @staticmethod
    def _from_dict(parsed: dict[str, Any]) -> tuple[str, list[Any]] | None:
        """Convert a decoded JSON object into ``(label, ids)``."""
        label = parsed.get("label")
        if not isinstance(label, str):
            return None
        ids = parsed.get("evidence_ids", [])
        if isinstance(ids, int):
            ids = [ids]
        if not isinstance(ids, list):
            ids = []
        return OutputParser._normalise(label), ids

    @staticmethod
    def _normalise(label: str) -> str:
        """Lower-case and squeeze whitespace out of a predicted label."""
        return " ".join(label.strip().lower().split())

    @staticmethod
    def _clean_ids(ids: list[Any], n_sentences: int) -> _CleanIds:
        """Keep integer IDs inside the note, de-duplicated and sorted."""
        kept: set[int] = set()
        dropped: list[int] = []
        for value in ids:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= number < n_sentences:
                kept.add(number)
            else:
                dropped.append(number)
        return _CleanIds(kept=tuple(sorted(kept)), dropped=tuple(sorted(dropped)))


@dataclass(frozen=True)
class _CleanIds:
    """Internal result of evidence-ID sanitising."""

    kept: tuple[int, ...]
    dropped: tuple[int, ...]
