"""Benchmarks for the assignment's three hand-written components.

Suites: ``decoding`` (uncached vs KV-cached), ``attention`` (Phase 5) and ``speculative``
(Phase 6). Only ``--smoke`` (tiny random model, tiny grid) is ever run by the agent; the real
grids run on Colab T4.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from transformers import PreTrainedModel, Qwen2Config, Qwen2ForCausalLM

from trialmatch import constants as c
from trialmatch.decoding import Decoder, DecoderFactory, UncachedDecoder
from trialmatch.modeling import LoadedModel, ModelLoader


class Statistics:
    """Small mean/std helper so the JSON schema stays uniform."""

    @staticmethod
    def summarise(values: Sequence[float]) -> dict[str, float]:
        """Return the mean and sample standard deviation (0 for a single value)."""
        return {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }


class DecodingBenchmark:
    """Compare uncached and KV-cached greedy decoding over an input/output length grid."""

    def __init__(
        self,
        model: PreTrainedModel,
        device: torch.device,
        vocab_size: int,
        config: c.BenchmarkConfig = c.BENCHMARK,
    ) -> None:
        """Store the model, device and grid configuration."""
        self._model = model
        self._device = device
        self._vocab_size = vocab_size
        self._config = config

    def run(self) -> dict[str, Any]:
        """Run every grid point and return the result document."""
        rows: list[dict[str, Any]] = []
        for input_len in self._config.decoding_input_lengths:
            for output_len in self._config.decoding_output_lengths:
                rows.append(self._measure(input_len, output_len))
        return {
            "suite": "decoding",
            "warmup_runs": self._config.warmup_runs,
            "timed_runs": self._config.timed_runs,
            "grid": {
                "input_lengths": list(self._config.decoding_input_lengths),
                "output_lengths": list(self._config.decoding_output_lengths),
            },
            "rows": rows,
        }

    def _measure(self, input_len: int, output_len: int) -> dict[str, Any]:
        """Time both decoders at one grid point."""
        generator = torch.Generator().manual_seed(c.SEED)
        ids = torch.randint(0, self._vocab_size, (input_len,), generator=generator).tolist()
        summary: dict[str, Any] = {"input_len": input_len, "output_len": output_len}
        tokens: dict[str, tuple[int, ...]] = {}
        for kind in ("uncached", "cached"):
            decoder = self._make_decoder(kind)
            reference = decoder.generate(ids, output_len)
            for _ in range(self._config.warmup_runs):
                decoder.generate(ids, output_len)
            runs = [decoder.generate(ids, output_len) for _ in range(self._config.timed_runs)]
            latencies = [run.latency_s for run in runs]
            mean_latency = statistics.fmean(latencies)
            tokens[kind] = reference.tokens
            summary[kind] = {
                "latency_s": Statistics.summarise(latencies),
                "prefill_s": statistics.fmean([run.prefill_s for run in runs]),
                "decode_s": statistics.fmean([run.decode_s for run in runs]),
                "tokens_per_s": len(reference.tokens) / mean_latency if mean_latency else 0.0,
                "peak_mem_bytes": max(run.peak_mem_bytes for run in runs),
                "n_forward_calls": reference.n_forward_calls,
            }
        summary["tokens_equal"] = tokens["uncached"] == tokens["cached"]
        summary["speedup"] = (
            summary["uncached"]["latency_s"]["mean"] / summary["cached"]["latency_s"]["mean"]
            if summary["cached"]["latency_s"]["mean"]
            else 0.0
        )
        return summary

    def _make_decoder(self, kind: str) -> Decoder:
        """Instantiate a decoder with no EOS stopping (fixed output length keeps the grid fair)."""
        if kind == "uncached":
            return UncachedDecoder(self._model, eos_token_ids=[], device=self._device)
        return DecoderFactory.create("cached", self._model, [], self._device)


class BenchmarkPlotter:
    """Figures for the benchmark suites."""

    @staticmethod
    def decoding(
        document: dict[str, Any], path: Path = c.FIGURES_DIR / "bench_decoding.pdf"
    ) -> None:
        """Plot mean latency and throughput against output length."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = document["grid"]
        figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
        for input_len in grid["input_lengths"]:
            rows = [r for r in document["rows"] if r["input_len"] == input_len]
            for kind, style in (("uncached", "--"), ("cached", "-")):
                xs = [r["output_len"] for r in rows]
                ys = [r[kind]["latency_s"]["mean"] for r in rows]
                left.plot(xs, ys, style, marker="o", label=f"{kind} · {input_len} in")
                right.plot(
                    xs,
                    [r[kind]["tokens_per_s"] for r in rows],
                    style,
                    marker="o",
                    label=f"{kind} · {input_len} in",
                )
        left.set_xlabel("new tokens")
        left.set_ylabel("latency (s)")
        left.set_title("Latency")
        right.set_xlabel("new tokens")
        right.set_ylabel("tokens / s")
        right.set_title("Throughput")
        for axis in (left, right):
            axis.grid(alpha=0.3)
        right.legend(fontsize=7, ncol=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)


class Benchmarker:
    """Load the model once and dispatch to the requested suite."""

    def __init__(self, model_name: str = c.BASE_MODEL_NAME, smoke: bool = False) -> None:
        """Store the model choice; ``smoke`` builds a tiny random model instead of downloading."""
        self._model_name = model_name
        self._smoke = smoke

    def run(self, suite: str) -> Path:
        """Run one suite, write ``results/bench_{suite}.json`` and return its path."""
        if suite != "decoding":
            raise ValueError(f"suite {suite!r} is not implemented yet")
        model, device, vocab, config = self._prepare()
        document = DecodingBenchmark(model, device, vocab, config).run()
        document["model"] = "tiny-random-qwen2" if self._smoke else self._model_name
        document["device"] = device.type
        # Smoke artefacts stay in the git-ignored checkpoint dir: results/ is for real runs.
        root = c.CHECKPOINTS_DIR / "smoke" if self._smoke else c.RESULTS_DIR
        path = root / f"bench_{suite}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        BenchmarkPlotter.decoding(document, root / "bench_decoding.pdf")
        return path

    def _prepare(self) -> tuple[PreTrainedModel, torch.device, int, c.BenchmarkConfig]:
        """Return model, device, vocabulary size and (possibly reduced) grid for the suite."""
        if self._smoke:
            return self._tiny_model()
        loaded: LoadedModel = ModelLoader(self._model_name, merge_adapter=True).load()
        return loaded.model, loaded.device, int(loaded.model.config.vocab_size), c.BENCHMARK

    @staticmethod
    def _tiny_model() -> tuple[PreTrainedModel, torch.device, int, c.BenchmarkConfig]:
        """Build a 2-layer random model and a tiny grid for the local smoke check."""
        torch.manual_seed(c.SEED)
        config = Qwen2Config(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            tie_word_embeddings=True,
        )
        # transformers' config-driven constructor is not annotated for mypy.
        model = Qwen2ForCausalLM(config)  # type: ignore[no-untyped-call]
        cast(Any, model).eval()
        tiny_grid = c.BenchmarkConfig(
            decoding_input_lengths=(16, 32),
            decoding_output_lengths=(4, 8),
            attention_lengths=(16, 32),
            speculative_gammas=(2,),
            warmup_runs=1,
            timed_runs=1,
        )
        return cast(PreTrainedModel, model), torch.device("cpu"), 256, tiny_grid


def main() -> None:
    """CLI: ``python -m trialmatch.benchmark --suite decoding [--smoke]``."""
    parser = argparse.ArgumentParser(
        description="Benchmark decoding, attention and speculative paths."
    )
    parser.add_argument(
        "--suite", choices=["decoding", "attention", "speculative"], default="decoding"
    )
    parser.add_argument("--model", default=c.BASE_MODEL_NAME)
    parser.add_argument("--smoke", action="store_true", help="tiny random model, tiny grid")
    args = parser.parse_args()
    path = Benchmarker(args.model, smoke=args.smoke).run(args.suite)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
