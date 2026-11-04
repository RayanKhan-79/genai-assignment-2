"""Patient-level screening pipeline and report export.

:class:`ScreeningPipeline` renders one prompt per criterion, calls the configured greedy decoder
and assembles the per-criterion predictions into a :class:`ScreeningResult`; the pure assembly step
is a separate static method so it can be tested without a model. :class:`ReportExporter` turns a
result into a Markdown or HTML report with the disclaimer, provenance and aggregation rule.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from trialmatch import constants as c
from trialmatch.data import (
    Example,
    PromptBuilder,
    PromptEncoder,
    PromptTooLongError,
    SentenceSplitter,
)
from trialmatch.evaluate import GenerationRunner
from trialmatch.modeling import LoadedModel
from trialmatch.schema import OutputParser


@dataclass(frozen=True)
class CriterionSpec:
    """One user-supplied criterion: its type and text.

    Raises:
        ValueError: If the type is neither ``inclusion`` nor ``exclusion``.
    """

    criterion_type: str
    text: str

    def __post_init__(self) -> None:
        """Reject unknown criterion types early, before any generation."""
        if self.criterion_type not in {
            c.CRITERION_TYPE_INCLUSION,
            c.CRITERION_TYPE_EXCLUSION,
        }:
            raise ValueError(f"unknown criterion type: {self.criterion_type!r}")
        if not self.text.strip():
            raise ValueError("criterion text must not be empty")


@dataclass(frozen=True)
class Provenance:
    """Everything the report needs to say where a prediction came from."""

    base_model: str
    adapter: str | None
    decode: str
    max_new_tokens: int
    created_at: str

    @staticmethod
    def now(base_model: str, adapter: str | None, decode: str, max_new_tokens: int) -> Provenance:
        """Build a provenance block stamped with the current UTC time."""
        return Provenance(
            base_model=base_model,
            adapter=adapter,
            decode=decode,
            max_new_tokens=max_new_tokens,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )


@dataclass(frozen=True)
class ScreeningItem:
    """One criterion's prediction, evidence quotes and explanation."""

    criterion_type: str
    criterion_text: str
    label: str
    evidence_ids: tuple[int, ...]
    evidence_text: tuple[str, ...]
    explanation: str
    latency_s: float
    raw: str
    strict: bool


@dataclass(frozen=True)
class ScreeningResult:
    """A complete screening run: the note, every criterion and the preliminary outcome."""

    note: str
    sentences: tuple[str, ...]
    items: tuple[ScreeningItem, ...]
    outcome: str
    missing_information: tuple[str, ...]
    provenance: Provenance


class ScreeningPipeline:
    """Screen one patient note against a list of criteria with a chosen decoder."""

    def __init__(
        self,
        loaded: LoadedModel,
        decode: str = "cached",
        max_new_tokens: int = c.GENERATION.max_new_tokens,
        draft: LoadedModel | None = None,
        builder: PromptBuilder | None = None,
        splitter: SentenceSplitter | None = None,
        parser: OutputParser | None = None,
    ) -> None:
        """Store the model bundle and the collaborating components.

        Args:
            loaded: Target model and tokenizer.
            decode: ``uncached``, ``cached`` or ``speculative``.
            max_new_tokens: Completion budget per criterion.
            draft: Draft model, required for ``speculative``.
            builder: Prompt renderer (defaults to the dataset one).
            splitter: Sentence splitter used for notes and evidence lookup.
            parser: Output parser (defaults to the tolerant one).
        """
        self._loaded = loaded
        self._builder = builder or PromptBuilder()
        self._splitter = splitter or SentenceSplitter()
        self._parser = parser or OutputParser()
        self._encoder = PromptEncoder(loaded.tokenizer, self._builder)
        self._runner = GenerationRunner(loaded, max_new_tokens, decode, draft)
        self._decode = decode
        self._max_new_tokens = max_new_tokens

    def screen(self, note: str, criteria: Sequence[CriterionSpec]) -> ScreeningResult:
        """Run every criterion and return the assembled result.

        Raises:
            ValueError: If the note has no sentences or no criteria were supplied.
            PromptTooLongError: If a rendered prompt exceeds ``MAX_SEQ_LEN``.
        """
        if not criteria:
            raise ValueError("add at least one criterion")
        if len(criteria) > c.APP_MAX_CRITERIA:
            raise ValueError(f"too many criteria: {len(criteria)} > {c.APP_MAX_CRITERIA}")
        sentences = self._splitter.split(note)
        if not sentences:
            raise ValueError("the patient note is empty")
        completions: list[tuple[str, float]] = []
        for spec in criteria:
            example = Example(
                system=c.SYSTEM_PROMPT,
                user=self._builder.user_prompt(sentences, spec.criterion_type, spec.text),
                target="",
            )
            if len(self._encoder.prompt_ids(example)) > c.GENERATION.max_seq_len:
                raise PromptTooLongError(
                    f"prompt exceeds {c.GENERATION.max_seq_len} tokens; shorten the note "
                    "or split it"
                )
            completions.append(self._runner.run(example))
        provenance = Provenance.now(
            self._loaded.base_name,
            self._loaded.adapter,
            self._decode,
            self._max_new_tokens,
        )
        return self.assemble(note, criteria, completions, provenance)

    @staticmethod
    def assemble(
        note: str,
        criteria: Sequence[CriterionSpec],
        completions: Sequence[tuple[str, float]],
        provenance: Provenance,
        parser: OutputParser | None = None,
        splitter: SentenceSplitter | None = None,
    ) -> ScreeningResult:
        """Turn raw completions into a result (pure; used by tests and the app)."""
        if len(completions) != len(criteria):
            raise ValueError("one completion per criterion is required")
        splitter = splitter or SentenceSplitter()
        parser = parser or OutputParser()
        sentences = splitter.split(note)
        items: list[ScreeningItem] = []
        for spec, (raw, latency) in zip(criteria, completions, strict=True):
            prediction = parser.parse(raw, spec.criterion_type, len(sentences))
            items.append(
                ScreeningItem(
                    criterion_type=spec.criterion_type,
                    criterion_text=spec.text,
                    label=prediction.label,
                    evidence_ids=prediction.evidence_ids,
                    evidence_text=tuple(sentences[i] for i in prediction.evidence_ids),
                    explanation=c.APP_EXPLANATIONS[prediction.label],
                    latency_s=latency,
                    raw=raw,
                    strict=prediction.strict,
                )
            )
        outcome = OverallAggregator.aggregate(
            [CriterionLabel(item.criterion_type, item.label) for item in items]
        )
        missing = tuple(
            f"{item.criterion_type}: {item.criterion_text} [{item.label}]"
            for item in items
            if item.label in {"not enough information", c.LABEL_INVALID}
        )
        return ScreeningResult(
            note=note,
            sentences=sentences,
            items=tuple(items),
            outcome=outcome,
            missing_information=missing,
            provenance=provenance,
        )


class ReportExporter:
    """Render a :class:`ScreeningResult` as Markdown and HTML."""

    @staticmethod
    def to_markdown(result: ScreeningResult) -> str:
        """Return the full Markdown report, disclaimer first."""
        provenance = result.provenance
        lines = [
            "# TrialMatch screening report",
            "",
            f"> **{c.APP_DISCLAIMER}**",
            "",
            f"- Generated: {provenance.created_at}",
            f"- Base model: `{provenance.base_model}`",
            f"- Adapter: `{provenance.adapter or 'none'}`",
            f"- Decoding: `{provenance.decode}`",
            f"- Generation budget: {provenance.max_new_tokens} new tokens",
            f"- Preliminary outcome: **{result.outcome}**",
            "",
            "## Aggregation rule",
            "",
            c.APP_AGGREGATION_RULE,
            "",
            "## Patient note",
            "",
        ]
        lines.extend(f"{index}. {text}" for index, text in enumerate(result.sentences))
        lines.extend(
            [
                "",
                "## Per-criterion results",
                "",
                "| # | Type | Criterion | Label | Evidence IDs | Evidence text | Explanation |",
                "|---|------|-----------|-------|--------------|---------------|-------------|",
            ]
        )
        for index, item in enumerate(result.items, start=1):
            lines.append(
                f"| {index} | {item.criterion_type} | {_cell(item.criterion_text)} | "
                f"{item.label} | {_ids(item.evidence_ids)} | "
                f"{_cell('; '.join(item.evidence_text))} | {_cell(item.explanation)} |"
            )
        lines.extend(["", "## Missing information", ""])
        if result.missing_information:
            lines.extend(f"- {_cell(entry)}" for entry in result.missing_information)
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Provenance",
                "",
                f"- Tool: {c.APP_TITLE}",
                f"- Base model: `{provenance.base_model}`",
                f"- Adapter: `{provenance.adapter or 'none'}`",
                f"- Decoder: `{provenance.decode}` (greedy, max {provenance.max_new_tokens} "
                "new tokens)",
                "- Labels are model predictions, not clinical decisions; evidence IDs refer to "
                "the numbered patient sentences above.",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def to_html(result: ScreeningResult) -> str:
        """Return a standalone HTML report with every value escaped."""
        provenance = result.provenance
        rows = "".join(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(item.criterion_type)}</td>"
            f"<td>{html.escape(item.criterion_text)}</td>"
            f"<td>{html.escape(item.label)}</td>"
            f"<td>{html.escape(_ids(item.evidence_ids))}</td>"
            f"<td>{html.escape('; '.join(item.evidence_text))}</td>"
            f"<td>{html.escape(item.explanation)}</td>"
            "</tr>"
            for index, item in enumerate(result.items, start=1)
        )
        sentences = "".join(f"<li>{html.escape(text)}</li>" for text in result.sentences)
        missing = (
            "<ul>"
            + "".join(f"<li>{html.escape(e)}</li>" for e in result.missing_information)
            + "</ul>"
            if result.missing_information
            else "<p>none</p>"
        )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TrialMatch screening report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
.disclaimer {{ background: #fff3cd; border: 1px solid #e0c068; padding: .75rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
th, td {{ border: 1px solid #ccc; padding: .35rem; vertical-align: top; text-align: left; }}
</style></head><body>
<h1>TrialMatch screening report</h1>
<p class="disclaimer"><strong>{html.escape(c.APP_DISCLAIMER)}</strong></p>
<ul>
<li>Generated: {html.escape(provenance.created_at)}</li>
<li>Base model: {html.escape(provenance.base_model)}</li>
<li>Adapter: {html.escape(provenance.adapter or "none")}</li>
<li>Decoding: {html.escape(provenance.decode)}</li>
<li>Preliminary outcome: <strong>{html.escape(result.outcome)}</strong></li>
</ul>
<h2>Aggregation rule</h2><p>{html.escape(c.APP_AGGREGATION_RULE)}</p>
<h2>Patient note</h2><ol>{sentences}</ol>
<h2>Per-criterion results</h2>
<table><thead><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th>
<th>Evidence IDs</th><th>Evidence text</th><th>Explanation</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Missing information</h2>{missing}
<h2>Provenance</h2>
<p>Labels are model predictions, not clinical decisions; evidence IDs refer to the numbered
patient sentences above. Generated by {html.escape(c.APP_TITLE)}.</p>
</body></html>
"""

    @staticmethod
    def write(result: ScreeningResult, directory: Path = c.APP_REPORT_DIR) -> tuple[Path, Path]:
        """Write both report formats and return ``(markdown_path, html_path)``."""
        directory.mkdir(parents=True, exist_ok=True)
        stamp = result.provenance.created_at.replace(":", "").replace("-", "")
        markdown_path = directory / f"screening_{stamp}.md"
        html_path = directory / f"screening_{stamp}.html"
        markdown_path.write_text(ReportExporter.to_markdown(result), encoding="utf-8")
        html_path.write_text(ReportExporter.to_html(result), encoding="utf-8")
        return markdown_path, html_path


def _cell(text: str) -> str:
    """Make a value safe inside a Markdown table row."""
    return text.replace("|", "\\|").replace("\n", " ")


def _ids(ids: Sequence[int]) -> str:
    """Render evidence IDs as a comma-separated list (empty when none)."""
    return ", ".join(str(value) for value in ids)


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
