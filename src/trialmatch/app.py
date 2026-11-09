"""Gradio application: patient screening, report download and a live decoder comparison.

The model loads lazily on first use (no globals); the disclaimer is part of the page header and of
every exported report. ``--smoke`` builds the UI and runs both tabs on the 0.5B base model without
opening a server, which is the local wiring check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

import gradio as gr
import torch
from huggingface_hub import snapshot_download

from trialmatch import constants as c
from trialmatch.data import Example, PromptBuilder, PromptEncoder, SentenceSplitter
from trialmatch.decoding import CachedDecoder, UncachedDecoder
from trialmatch.modeling import LoadedModel, ModelLoader
from trialmatch.pipeline import CriterionSpec, ReportExporter, ScreeningPipeline, ScreeningResult
from trialmatch.speculative import SpeculativeDecoder

TABLE_HEADERS = [
    "#",
    "Type",
    "Criterion",
    "Label",
    "Evidence IDs",
    "Evidence text",
    "Explanation",
]
BENCH_HEADERS = [
    "Decoder",
    "Latency (s)",
    "Tokens/s",
    "Forward calls",
    "Peak memory (MiB)",
    "Acceptance rate",
    "Matches cached target",
]


def _force_agg_backend() -> None:
    """Point matplotlib at a static backend before Gradio imports it lazily.

    Colab exports ``MPLBACKEND=module://matplotlib_inline.backend_inline``, but that module is not
    installed in this virtual environment. Gradio imports matplotlib on the first request (and
    restores the backend afterwards), so the invalid value must be replaced before the server runs.
    """
    os.environ["MPLBACKEND"] = c.MPL_BACKEND


def _failure(prefix: str, error: Exception) -> str:
    """Format a UI failure, adding a fallback hint when the accelerator is out of memory."""
    message = f"{prefix}: {error}"
    lowered = str(error).lower()
    if any(token in lowered for token in ("out of memory", "outofmemory", "insufficient memory")):
        message += (
            " — the GPU is out of memory. Restart with "
            "`uv run python -m trialmatch.app --device cpu`: the 1.5B model in fp32 (about 6 GB) "
            "plus activations and the KV cache does not fit an 8 GB Apple GPU's working set."
        )
    return message


class ScreeningApp:
    """Owns the lazily loaded models, the pipelines and the Gradio Blocks layout."""

    def __init__(
        self,
        model_name: str = c.BASE_MODEL_NAME,
        adapter: str | Path | None = None,
        device: torch.device | None = None,
        smoke: bool = False,
        max_new_tokens: int = c.GENERATION.max_new_tokens,
    ) -> None:
        """Configure the app; no weights are read until the first request."""
        self._model_name = model_name
        self._adapter = adapter
        self._device = device
        self._smoke = smoke
        self._max_new_tokens = max_new_tokens
        self._loaded: LoadedModel | None = None
        self._draft: LoadedModel | None = None
        self._pipelines: dict[str, ScreeningPipeline] = {}
        self._notice = ""

    # ------------------------------------------------------------------ model loading
    def _adapter_source(self) -> str | None:
        """Return a local adapter path, downloading it from the Hub once if necessary."""
        if self._smoke:
            return None
        if self._adapter is not None:
            return str(self._adapter)
        if c.ADAPTER_DIR.exists():
            return str(c.ADAPTER_DIR)
        try:
            snapshot_download(repo_id=c.ADAPTER_REPO, local_dir=c.ADAPTER_DIR)
        except Exception:  # offline, private or not uploaded yet: base model still works
            return None
        return str(c.ADAPTER_DIR)

    def _target(self) -> LoadedModel:
        """Load the target model once (merged adapter, so every decoder can use it)."""
        if self._loaded is None:
            source = self._adapter_source()
            loaded = ModelLoader(
                self._model_name,
                adapter=source,
                merge_adapter=True,
                device=self._device,
            ).load()
            self._loaded = loaded
            self._notice = (
                ""
                if source or self._smoke
                else "No fine-tuned adapter found; using the base model."
            )
        return self._loaded

    def _draft_model(self) -> LoadedModel:
        """Load the draft model once, on the same device as the target."""
        if self._draft is None:
            self._draft = ModelLoader(c.DRAFT_MODEL_NAME, device=self._target().device).load()
        return self._draft

    def _pipeline(self, decode: str) -> ScreeningPipeline:
        """Return (and cache) the screening pipeline for one decoding mode."""
        if decode not in c.APP_DECODE_MODES:
            raise ValueError(f"unknown decoding mode: {decode!r}")
        if decode not in self._pipelines:
            draft = self._draft_model() if decode == "speculative" else None
            self._pipelines[decode] = ScreeningPipeline(
                self._target(),
                decode=decode,
                max_new_tokens=self._max_new_tokens,
                draft=draft,
            )
        return self._pipelines[decode]

    # ------------------------------------------------------------------ screening tab
    @staticmethod
    def _criteria(inclusion: str, exclusion: str) -> list[CriterionSpec]:
        """Parse the two textboxes (one criterion per line) into specs."""
        specs = [
            CriterionSpec(c.CRITERION_TYPE_INCLUSION, line.strip())
            for line in inclusion.splitlines()
            if line.strip()
        ]
        specs.extend(
            CriterionSpec(c.CRITERION_TYPE_EXCLUSION, line.strip())
            for line in exclusion.splitlines()
            if line.strip()
        )
        if not specs:
            raise ValueError("add at least one criterion (one per line)")
        if len(specs) > c.APP_MAX_CRITERIA:
            raise ValueError(f"too many criteria: {len(specs)} > {c.APP_MAX_CRITERIA}")
        return specs

    def screen(
        self, note: str, inclusion: str, exclusion: str, decode_label: str
    ) -> tuple[list[list[Any]], str, str, str, str | None, str]:
        """Run the screening tab; every failure becomes a readable status message."""
        try:
            decode = c.APP_DECODE_LABELS[decode_label]
            result = self._pipeline(decode).screen(note, self._criteria(inclusion, exclusion))
            report, _ = ReportExporter.write(result)
            elapsed = sum(item.latency_s for item in result.items)
            status = (
                f"Screened {len(result.items)} criteria with {decode_label} decoding in "
                f"{elapsed:.2f} s."
            )
            if self._notice:
                status = f"{status} {self._notice}"
            return (
                _result_rows(result),
                _outcome_markdown(result),
                _missing_markdown(result),
                _raw_markdown(result),
                str(report),
                status,
            )
        except Exception as error:  # UI boundary: never show a stack trace
            return [], "", "", "", None, _failure("Could not screen", error)

    # ------------------------------------------------------------------ benchmark tab
    def benchmark(
        self, note: str, inclusion: str, exclusion: str
    ) -> tuple[list[list[Any]], Any, str]:
        """Time the three decoders on the first criterion of the current input."""
        try:
            specs = self._criteria(inclusion, exclusion)
            sentences = SentenceSplitter().split(note)
            if not sentences:
                raise ValueError("the patient note is empty")
            target = self._target()
            builder = PromptBuilder()
            user = builder.user_prompt(sentences, specs[0].criterion_type, specs[0].text)
            example = Example(system=c.SYSTEM_PROMPT, user=user, target="")
            prompt_ids = PromptEncoder(target.tokenizer).prompt_ids(example)
            budget = min(self._max_new_tokens, c.APP_BENCHMARK_MAX_NEW_TOKENS)
            eos = target.tokenizer.eos_token_id
            eos_ids = [eos] if eos is not None else []

            uncached = UncachedDecoder(target.model, eos_ids, target.device).generate(
                prompt_ids, budget
            )
            decoders = {
                "Uncached": lambda: UncachedDecoder(target.model, eos_ids, target.device).generate(
                    prompt_ids, budget
                ),
                "KV cache": lambda: CachedDecoder(target.model, eos_ids, target.device).generate(
                    prompt_ids, budget
                ),
                "Speculative": lambda: SpeculativeDecoder(
                    target.model,
                    self._draft_model().model,
                    eos_ids,
                    target.device,
                    gamma=c.APP_SPECULATIVE_GAMMA,
                    tokenizers=(target.tokenizer, self._draft_model().tokenizer),
                ).generate(prompt_ids, budget),
            }
            rows: list[list[Any]] = []
            reference = uncached.tokens
            for name, run in decoders.items():
                result = run()
                acceptance = (
                    f"{result.stats.acceptance_rate:.0%}" if result.stats is not None else "—"
                )
                rows.append(
                    [
                        name,
                        round(result.latency_s, 3),
                        round(len(result.tokens) / result.latency_s, 2)
                        if result.latency_s
                        else 0.0,
                        result.n_forward_calls,
                        round(result.peak_mem_bytes / 2**20, 2),
                        acceptance,
                        "yes" if result.tokens == reference else "no",
                    ]
                )
            figure = _benchmark_figure(rows)
            return (
                rows,
                figure,
                (
                    f"Compared three decoders on one prompt ({len(prompt_ids)} prompt tokens, "
                    f"{budget} new tokens)."
                ),
            )
        except Exception as error:  # UI boundary
            return [], None, _failure("Could not benchmark", error)

    def view_benchmark(self, name: str) -> str:
        """Render a stored ``results/bench_*.json`` document as Markdown."""
        path = c.RESULTS_DIR / name
        if not name or not path.exists():
            return "No stored benchmark file selected."
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return f"Could not read {name}: {error}"
        return _benchmark_markdown(document, name)

    @staticmethod
    def _benchmark_files() -> list[str]:
        """List the benchmark files currently present in ``results/``."""
        if not c.RESULTS_DIR.exists():
            return []
        return sorted(path.name for path in c.RESULTS_DIR.glob("bench_*.json"))

    def _read_upload(self, path: str | None) -> str:
        """Read an uploaded ``.txt`` note into the textbox (empty on failure)."""
        if not path:
            return ""
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    # ------------------------------------------------------------------ layout
    def build(self) -> gr.Blocks:
        """Build the Blocks layout (components only; model loading stays lazy)."""
        # Gradio wraps every request in a matplotlib backend manager; fix MPLBACKEND first.
        _force_agg_backend()
        with gr.Blocks(title=c.APP_TITLE) as demo:
            gr.Markdown(f"# {c.APP_TITLE}")
            gr.Markdown(f"**{c.APP_DISCLAIMER}**")
            with gr.Tabs():
                with gr.Tab("Screening"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            note = gr.Textbox(
                                label="Patient note",
                                lines=14,
                                placeholder="0. 68-year-old woman with ...",
                            )
                            upload = gr.File(
                                label="... or upload a .txt note",
                                file_types=[".txt"],
                                type="filepath",
                            )
                            inclusion = gr.Textbox(
                                label="Inclusion criteria (one per line)", lines=4
                            )
                            exclusion = gr.Textbox(
                                label="Exclusion criteria (one per line)", lines=4
                            )
                            decode = gr.Dropdown(
                                choices=list(c.APP_DECODE_LABELS),
                                value="KV cache",
                                label="Decoding",
                            )
                            run_button = gr.Button("Run screening", variant="primary")
                        with gr.Column(scale=4):
                            status = gr.Markdown(
                                "Ready. The model loads on first use; the first run is slow."
                            )
                            outcome = gr.Markdown()
                            table = gr.Dataframe(
                                headers=TABLE_HEADERS,
                                datatype=["number", "str", "str", "str", "str", "str", "str"],
                                wrap=True,
                                interactive=False,
                                label="Per-criterion results",
                            )
                            missing = gr.Markdown()
                            raw = gr.Markdown()
                            download = gr.DownloadButton("Download Markdown report")
                    upload.change(self._read_upload, inputs=upload, outputs=note)
                    run_button.click(
                        self.screen,
                        inputs=[note, inclusion, exclusion, decode],
                        outputs=[table, outcome, missing, raw, download, status],
                    )
                with gr.Tab("Benchmark"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            bench_note = gr.Textbox(
                                label="Patient note", lines=10, placeholder="0. ..."
                            )
                            bench_inclusion = gr.Textbox(
                                label="Inclusion criteria (one per line)", lines=3
                            )
                            bench_exclusion = gr.Textbox(
                                label="Exclusion criteria (one per line)", lines=3
                            )
                            bench_button = gr.Button(
                                "Run comparison on this input", variant="primary"
                            )
                        with gr.Column(scale=4):
                            bench_status = gr.Markdown()
                            bench_table = gr.Dataframe(
                                headers=BENCH_HEADERS,
                                datatype=[
                                    "str",
                                    "number",
                                    "number",
                                    "number",
                                    "number",
                                    "str",
                                    "str",
                                ],
                                wrap=True,
                                interactive=False,
                                label="Live decoder comparison",
                            )
                            bench_plot = gr.Plot(label="Latency and throughput")
                    with gr.Accordion("Stored Colab benchmark files", open=False):
                        bench_file = gr.Dropdown(
                            choices=self._benchmark_files(),
                            label="results/bench_*.json",
                        )
                        bench_view = gr.Markdown(
                            "Run job 5 on Colab and push `results/` to populate this viewer."
                        )
                        bench_file.change(
                            self.view_benchmark, inputs=bench_file, outputs=bench_view
                        )
                    bench_button.click(
                        self.benchmark,
                        inputs=[bench_note, bench_inclusion, bench_exclusion],
                        outputs=[bench_table, bench_plot, bench_status],
                    )
        return cast(gr.Blocks, demo)

    def run_smoke(self) -> None:
        """Exercise both tabs on the tiny/small model without opening a server."""
        demo = self.build()
        demo.get_config_file()  # raises if the layout is invalid
        note = (
            "0. The patient is a 60-year-old woman.\n1. She has diabetes.\n2. She takes metformin."
        )
        inclusion = "Diagnosed with diabetes"
        exclusion = "Pregnant"
        rows, outcome, missing, raw, report, status = self.screen(
            note, inclusion, exclusion, "KV cache"
        )
        if not rows or not report or Path(report).suffix != ".md":
            raise RuntimeError(f"screen smoke failed: {status}")
        bench_rows, figure, bench_status = self.benchmark(note, inclusion, exclusion)
        if not bench_rows or figure is None:
            raise RuntimeError(f"benchmark smoke failed: {bench_status}")
        print("UI smoke ok")
        print(f"  status: {status}")
        print(f"  rows: {len(rows)}, outcome: {outcome.splitlines()[0]}")
        print(f"  missing: {missing.splitlines()[0] if missing else 'none'}")
        print(f"  benchmark: {bench_status} rows={len(bench_rows)} raw={raw!r}")


def _result_rows(result: ScreeningResult) -> list[list[Any]]:
    """Convert a screening result into table rows."""
    return [
        [
            index,
            item.criterion_type,
            item.criterion_text,
            item.label,
            ", ".join(str(value) for value in item.evidence_ids),
            "; ".join(item.evidence_text),
            item.explanation,
        ]
        for index, item in enumerate(result.items, start=1)
    ]


def _outcome_markdown(result: ScreeningResult) -> str:
    """Outcome heading plus the documented aggregation rule."""
    return f"### Preliminary outcome: {result.outcome}\n\n{c.APP_AGGREGATION_RULE}"


def _missing_markdown(result: ScreeningResult) -> str:
    """Missing-information bullet list (or an explicit 'none')."""
    if not result.missing_information:
        return "**Missing information:** none."
    bullets = "\n".join(f"- {entry}" for entry in result.missing_information)
    return f"**Missing information:**\n\n{bullets}"


def _raw_markdown(result: ScreeningResult) -> str:
    """Show raw completions only for predictions that needed repair or failed to parse."""
    lines = [
        f"- **{item.criterion_type} / {item.label}**: `{item.raw.strip()[:300]}`"
        for item in result.items
        if not item.strict
    ]
    if not lines:
        return ""
    return "**Unparsed or repaired outputs:**\n\n" + "\n".join(lines)


def _benchmark_figure(rows: list[list[Any]]) -> Any:
    """Build the two-panel latency/throughput figure for the benchmark tab."""
    _force_agg_backend()
    import matplotlib

    matplotlib.use(c.MPL_BACKEND)
    import matplotlib.pyplot as plt

    names = [row[0] for row in rows]
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
    left.bar(names, [row[1] for row in rows], color="#4c72b0")
    left.set_ylabel("latency (s)")
    left.set_title("Latency on this prompt")
    right.bar(names, [row[2] for row in rows], color="#55a868")
    right.set_ylabel("tokens / s")
    right.set_title("Throughput")
    for axis in (left, right):
        axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    return figure


def _benchmark_markdown(document: dict[str, Any], name: str) -> str:
    """Render a stored benchmark JSON document as a compact Markdown summary."""
    suite = document.get("suite", "unknown")
    lines = [f"### `{name}`", "", f"- Suite: **{suite}**"]
    for key in ("model", "target_model", "draft_model", "device", "dtype"):
        if key in document:
            lines.append(f"- {key}: `{document[key]}`")
    lines.append("")
    if suite == "speculative":
        lines.append(
            f"Baseline cached target: {document['baseline']['latency_s']['mean']:.3f} s "
            f"({document['baseline']['tokens_per_s']:.1f} tokens/s)"
        )
        lines.extend(
            [
                "",
                "| Gamma | Acceptance | Speedup | Tokens equal |",
                "|-------|------------|---------|--------------|",
            ]
        )
        lines.extend(
            f"| {row['gamma']} | {row['acceptance_rate']:.1%} | {row['speedup']:.2f}x | "
            f"{row['tokens_equal']} |"
            for row in document["rows"]
        )
    elif suite == "decoding":
        lines.extend(
            [
                "| Input | Output | Cached (s) | Speedup | Tokens equal |",
                "|-------|--------|------------|---------|--------------|",
            ]
        )
        lines.extend(
            f"| {row['input_len']} | {row['output_len']} | "
            f"{row['cached']['latency_s']['mean']:.4f} | {row['speedup']:.2f}x | "
            f"{row['tokens_equal']} |"
            for row in document["rows"]
        )
    elif suite == "attention":
        lines.extend(
            [
                "| Length | Variant | Latency (s) | Max abs err |",
                "|--------|---------|-------------|-------------|",
            ]
        )
        for row in document["rows"]:
            for variant, payload in row["variants"].items():
                if not payload.get("supported"):
                    continue
                lines.append(
                    f"| {row['length']} | {variant} | "
                    f"{payload['latency_s']['mean']:.4f} | {payload['max_abs_err']:.2e} |"
                )
    else:
        lines.append("```json\n" + json.dumps(document, indent=2)[:2000] + "\n```")
    return "\n".join(lines)


def main() -> None:
    """CLI: ``uv run python -m trialmatch.app [--smoke] [--device cpu|mps|cuda]``."""
    parser = argparse.ArgumentParser(description="Run the TrialMatch screening app.")
    parser.add_argument("--smoke", action="store_true", help="UI wiring check, no server")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=c.GENERATION.max_new_tokens)
    args = parser.parse_args()
    device = None if args.device == "auto" else torch.device(args.device)
    app = ScreeningApp(
        model_name=c.SMOKE_MODEL_NAME if args.smoke else c.BASE_MODEL_NAME,
        adapter=args.adapter,
        device=device,
        smoke=args.smoke,
        max_new_tokens=64 if args.smoke else args.max_new_tokens,
    )
    if args.smoke:
        app.run_smoke()
        return
    app.build().launch()


if __name__ == "__main__":
    main()
