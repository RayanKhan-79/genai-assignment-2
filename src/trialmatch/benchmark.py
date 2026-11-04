"""Benchmarks for the assignment's three hand-written components.

Suites: ``decoding`` (uncached vs KV-cached), ``attention`` (Phase 5) and ``speculative``
(Phase 6). Only ``--smoke`` (tiny random model, tiny grid) is ever run by the agent; the real
grids run on Colab T4.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from trialmatch import constants as c
from trialmatch.attention import ReferenceAttention, TiledAttention
from trialmatch.data import AnnotationLoader, PromptBuilder, PromptEncoder, SplitManager
from trialmatch.decoding import CachedDecoder, Decoder, DecoderFactory, Timing, UncachedDecoder
from trialmatch.modeling import LoadedModel, ModelLoader
from trialmatch.speculative import SpeculativeDecoder


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


class AttentionBenchmark:
    """Compare reference, tiled and PyTorch SDPA attention over sequence lengths.

    Errors are measured against the reference implementation computed in fp32, which is the
    mathematical definition of the operation; the reference itself is therefore exactly zero.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        config: c.BenchmarkConfig = c.BENCHMARK,
    ) -> None:
        """Store the device, compute dtype and grid configuration."""
        self._device = device
        self._dtype = dtype
        self._config = config

    def run(self) -> dict[str, Any]:
        """Run every sequence length and return the result document."""
        rows = [self._measure(length) for length in self._config.attention_lengths]
        return {
            "suite": "attention",
            "device": self._device.type,
            "dtype": str(self._dtype).replace("torch.", ""),
            "shape": {
                "batch": self._config.attention_batch,
                "heads": self._config.attention_heads,
                "head_dim": self._config.attention_head_dim,
                "causal": True,
            },
            "block": {"q": self._config.attention_block_q, "k": self._config.attention_block_k},
            "warmup_runs": self._config.warmup_runs,
            "timed_runs": self._config.timed_runs,
            "rows": rows,
        }

    def _inputs(self, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return random Q/K/V of the configured shape on the benchmark device."""
        generator = torch.Generator().manual_seed(c.SEED)
        shape = (
            self._config.attention_batch,
            self._config.attention_heads,
            length,
            self._config.attention_head_dim,
        )
        tensors = [
            torch.randn(shape, generator=generator).to(self._device).to(self._dtype)
            for _ in range(3)
        ]
        return tensors[0], tensors[1], tensors[2]

    def _measure(self, length: int) -> dict[str, Any]:
        """Time every variant at one sequence length and compare it with the reference."""
        query, key, value = self._inputs(length)
        with torch.inference_mode():
            truth = ReferenceAttention.forward(
                query.float(), key.float(), value.float(), causal=True
            )
        variants: dict[str, Any] = {}
        for name, call in self._variants(query, key, value):
            variants[name] = self._time(name, call, truth)
        return {"length": length, "variants": variants}

    def _variants(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> list[tuple[str, Any]]:
        """Return ``(name, callable)`` pairs for every attention implementation."""
        tiled = TiledAttention(self._config.attention_block_q, self._config.attention_block_k)
        variants: list[tuple[str, Any]] = [
            ("reference", lambda: ReferenceAttention.forward(query, key, value, causal=True)),
            ("tiled", lambda: tiled.forward(query, key, value, causal=True)),
        ]
        for backend, label in (
            ("MATH", "sdpa_math"),
            ("EFFICIENT_ATTENTION", "sdpa_efficient"),
            ("FLASH_ATTENTION", "sdpa_flash"),
        ):
            variants.append(
                (label, self._sdpa_call(query, key, value, getattr(SDPBackend, backend)))
            )
        return variants

    @staticmethod
    def _sdpa_call(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, backend: SDPBackend
    ) -> Any:
        """Wrap ``scaled_dot_product_attention`` in an explicit backend selection."""

        def call() -> torch.Tensor:
            with sdpa_kernel(backend):
                return scaled_dot_product_attention(query, key, value, is_causal=True)

        return call

    def _time(self, name: str, call: Any, truth: torch.Tensor) -> dict[str, Any]:
        """Time one variant, returning ``supported=False`` if the backend refuses to run."""
        for _ in range(self._config.warmup_runs):
            try:
                call()
            except RuntimeError:
                return {"supported": False, "reason": "backend unavailable on this device"}
        latencies: list[float] = []
        peak = 0
        error_max = 0.0
        error_mean = 0.0
        for _ in range(self._config.timed_runs):
            Timing.reset_peak_memory()
            Timing.sync()
            start = time.perf_counter()
            with torch.inference_mode():
                result = call()
            Timing.sync()
            latencies.append(time.perf_counter() - start)
            peak = max(peak, Timing.peak_memory_bytes())
            difference = (result.float() - truth).abs()
            error_max = max(error_max, float(difference.max()))
            error_mean = max(error_mean, float(difference.mean()))
        return {
            "supported": True,
            "latency_s": Statistics.summarise(latencies),
            "peak_mem_bytes": peak,
            "max_abs_err": error_max,
            "mean_abs_err": error_mean,
        }


class AttentionPlotter:
    """Runtime and memory figures for the attention benchmark."""

    @staticmethod
    def plot(document: dict[str, Any], path: Path = c.FIGURES_DIR / "bench_attention.pdf") -> None:
        """Write latency (log-log) and peak memory against sequence length."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
        names = list(document["rows"][0]["variants"]) if document["rows"] else []
        for name in names:
            supported = [row for row in document["rows"] if row["variants"][name]["supported"]]
            lengths = [row["length"] for row in supported]
            left.plot(
                lengths,
                [row["variants"][name]["latency_s"]["mean"] for row in supported],
                marker="o",
                label=name,
            )
            right.plot(
                lengths,
                [row["variants"][name]["peak_mem_bytes"] / 2**20 for row in supported],
                marker="o",
                label=name,
            )
        left.set_xlabel("sequence length")
        left.set_ylabel("latency (s)")
        left.set_title("Attention latency")
        left.set_xscale("log", base=2)
        left.set_yscale("log")
        right.set_xlabel("sequence length")
        right.set_ylabel("peak memory (MiB)")
        right.set_title("Peak allocated memory")
        right.set_xscale("log", base=2)
        for axis in (left, right):
            axis.grid(alpha=0.3, which="both")
        left.legend(fontsize=7)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)


class SpeculativeBenchmark:
    """Compare KV-cached target decoding with speculative decoding for several gammas.

    Every configuration decodes a fixed token budget with no EOS stop, so throughput is measured on
    exactly the same token work; the cached target output is the ground truth and any divergence is
    reported rather than hidden. One prompt is also decoded with Hugging Face's assisted generation
    (``assistant_model``) as an independent cross-check of the hand-written draft/verify loop.
    """

    def __init__(
        self,
        target: PreTrainedModel,
        draft: PreTrainedModel,
        device: torch.device,
        prompts: Sequence[Sequence[int]],
        config: c.BenchmarkConfig,
        max_new_tokens: int,
        tokenizers: tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase] | None = None,
        target_label: str = "target",
        draft_label: str = "draft",
    ) -> None:
        """Store both models, the validation prompts and the gamma grid."""
        if not prompts:
            raise ValueError("speculative benchmark needs at least one prompt")
        self._target = target
        self._draft = draft
        self._device = device
        self._prompts = prompts
        self._config = config
        self._max_new_tokens = max_new_tokens
        self._tokenizers = tokenizers
        self._target_label = target_label
        self._draft_label = draft_label

    def run(self) -> dict[str, Any]:
        """Time the cached baseline and every gamma, then return the result document."""
        baseline = CachedDecoder(self._target, eos_token_ids=[], device=self._device)
        for _ in range(self._config.warmup_runs):
            baseline.generate(self._prompts[0], self._max_new_tokens)
        baseline_runs = [baseline.generate(ids, self._max_new_tokens) for ids in self._prompts]
        baseline_tokens = [run.tokens for run in baseline_runs]
        baseline_latency = sum(run.latency_s for run in baseline_runs)
        baseline_document = {
            "latency_s": Statistics.summarise([run.latency_s for run in baseline_runs]),
            "total_s": baseline_latency,
            "total_tokens": sum(len(run.tokens) for run in baseline_runs),
            "tokens_per_s": (
                sum(len(run.tokens) for run in baseline_runs) / baseline_latency
                if baseline_latency
                else 0.0
            ),
        }
        rows = [
            self._measure(gamma, baseline_tokens, baseline_latency)
            for gamma in self._config.speculative_gammas
        ]
        return {
            "suite": "speculative",
            "target_model": self._target_label,
            "draft_model": self._draft_label,
            "device": self._device.type,
            "dtype": str(next(self._target.parameters()).dtype).replace("torch.", ""),
            "n_prompts": len(self._prompts),
            "max_new_tokens": self._max_new_tokens,
            "gammas": list(self._config.speculative_gammas),
            "warmup_runs": self._config.warmup_runs,
            "baseline": baseline_document,
            "rows": rows,
            "reference_check": self._reference_check(self._prompts[0], baseline_tokens[0]),
        }

    def _measure(
        self, gamma: int, baseline_tokens: Sequence[tuple[int, ...]], baseline_s: float
    ) -> dict[str, Any]:
        """Run one gamma over every prompt and aggregate acceptance, latency and equality."""
        decoder = SpeculativeDecoder(
            self._target,
            self._draft,
            eos_token_ids=[],
            device=self._device,
            gamma=gamma,
            tokenizers=self._tokenizers,
        )
        for _ in range(self._config.warmup_runs):
            decoder.generate(self._prompts[0], self._max_new_tokens)
        runs = [decoder.generate(ids, self._max_new_tokens) for ids in self._prompts]
        stats = [run.stats for run in runs]
        if any(entry is None for entry in stats):
            raise ValueError("speculative decoder did not report statistics")
        drafted = sum(entry.drafted for entry in stats if entry is not None)
        accepted = sum(entry.accepted for entry in stats if entry is not None)
        rounds = sum(entry.rounds for entry in stats if entry is not None)
        total_s = sum(run.latency_s for run in runs)
        tokens = sum(len(run.tokens) for run in runs)
        mismatches = sum(
            run.tokens != reference for run, reference in zip(runs, baseline_tokens, strict=True)
        )
        return {
            "gamma": gamma,
            "latency_s": Statistics.summarise([run.latency_s for run in runs]),
            "total_s": total_s,
            "total_tokens": tokens,
            "tokens_per_s": tokens / total_s if total_s else 0.0,
            "peak_mem_bytes": max(run.peak_mem_bytes for run in runs),
            "acceptance_rate": accepted / drafted if drafted else 0.0,
            "mean_accepted_per_round": accepted / rounds if rounds else 0.0,
            "rounds": rounds,
            "target_forward_calls": sum(run.n_forward_calls for run in runs),
            "draft_forward_calls": sum(
                entry.draft_forward_calls for entry in stats if entry is not None
            ),
            "tokens_equal": mismatches == 0,
            "mismatched_prompts": mismatches,
            "speedup": baseline_s / total_s if total_s else 0.0,
        }

    def _reference_check(
        self, ids: Sequence[int], baseline_tokens: tuple[int, ...]
    ) -> dict[str, Any]:
        """Cross-check one prompt against ``generate(assistant_model=draft)``.

        Hugging Face stops at EOS while the benchmark decodes a fixed budget, so the comparison is
        on the common prefix; a mismatch there would mean the hand-written loop is not greedy.
        """
        inputs = torch.tensor([list(ids)], device=self._device)
        try:
            with torch.inference_mode():
                output = cast(Any, self._target).generate(
                    inputs,
                    assistant_model=self._draft,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    pad_token_id=0,
                )
        except (RuntimeError, ValueError) as error:  # backend or API unavailable
            return {"supported": False, "reason": str(error)[:200]}
        tokens = tuple(int(token) for token in output[0, inputs.shape[1] :].tolist())
        prefix = min(len(tokens), len(baseline_tokens))
        return {
            "supported": True,
            "hf_tokens": len(tokens),
            "prefix_length": prefix,
            "matches_prefix": tokens[:prefix] == baseline_tokens[:prefix],
        }


class SpeculativePlotter:
    """Latency, throughput and acceptance figures for the speculative benchmark."""

    @staticmethod
    def plot(
        document: dict[str, Any], path: Path = c.FIGURES_DIR / "bench_speculative.pdf"
    ) -> None:
        """Draw latency and acceptance rate per gamma against the cached baseline."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = document["rows"]
        labels = [f"gamma={row['gamma']}" for row in rows]
        figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
        left.bar(labels, [row["latency_s"]["mean"] for row in rows], color="#4c72b0")
        left.axhline(
            document["baseline"]["latency_s"]["mean"],
            linestyle="--",
            color="black",
            label="cached target",
        )
        left.set_ylabel("latency per prompt (s)")
        left.set_title("Speculative decoding latency")
        left.legend(fontsize=8)
        right.bar(labels, [100.0 * row["acceptance_rate"] for row in rows], color="#55a868")
        for index, row in enumerate(rows):
            right.text(
                index,
                100.0 * row["acceptance_rate"],
                f"{row['speedup']:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        right.set_ylabel("accepted drafts (%)")
        right.set_title("Acceptance rate (speedup vs cached)")
        for axis in (left, right):
            axis.grid(alpha=0.3, axis="y")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)


class Benchmarker:
    """Load the model once and dispatch to the requested suite."""

    def __init__(
        self,
        model_name: str = c.BASE_MODEL_NAME,
        smoke: bool = False,
        target_tag: str = c.MODEL_TAG_BASE,
        adapter: str | Path | None = None,
        prompts: int | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        """Store the model choices; ``smoke`` builds tiny random models instead of downloading."""
        self._model_name = model_name
        self._smoke = smoke
        self._target_tag = target_tag
        self._adapter = adapter
        self._prompts = prompts or c.BENCHMARK.speculative_prompts
        self._max_new_tokens = max_new_tokens or c.GENERATION.max_new_tokens

    def run(self, suite: str) -> Path:
        """Run one suite, write ``results/bench_{suite}.json`` and return its path."""
        if suite == "attention":
            document = AttentionBenchmark(*self._attention_setup()).run()
            AttentionPlotter.plot(document, self._figure_root() / "bench_attention.pdf")
        elif suite == "decoding":
            model, device, vocab, config = self._prepare()
            document = DecodingBenchmark(model, device, vocab, config).run()
            document["model"] = "tiny-random-qwen2" if self._smoke else self._model_name
            document["device"] = device.type
            BenchmarkPlotter.decoding(document, self._figure_root() / "bench_decoding.pdf")
        elif suite == "speculative":
            document = self._speculative_setup().run()
            SpeculativePlotter.plot(document, self._figure_root() / "bench_speculative.pdf")
        else:
            raise ValueError(f"suite {suite!r} is not implemented yet")
        path = self._figure_root() / f"bench_{suite}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    def _figure_root(self) -> Path:
        """Smoke artefacts stay in the git-ignored checkpoint dir; results/ is for real runs."""
        return c.CHECKPOINTS_DIR / "smoke" if self._smoke else c.RESULTS_DIR

    def _attention_setup(self) -> tuple[torch.device, torch.dtype, c.BenchmarkConfig]:
        """Attention needs no weights, only a device, dtype and (tiny) grid."""
        if self._smoke:
            _, device, _, _ = self._tiny_model()
            dtype = torch.float32
            tiny = c.BenchmarkConfig(
                attention_lengths=(64, 96),
                attention_block_q=32,
                attention_block_k=32,
                warmup_runs=1,
                timed_runs=1,
            )
            return device, dtype, tiny
        device, dtype = ModelLoader.resolve_device()
        return device, dtype, c.BENCHMARK

    def _prepare(self) -> tuple[PreTrainedModel, torch.device, int, c.BenchmarkConfig]:
        """Return model, device, vocabulary size and (possibly reduced) grid for the suite."""
        if self._smoke:
            return self._tiny_model()
        loaded: LoadedModel = ModelLoader(
            self._model_name, adapter=self._adapter, merge_adapter=True
        ).load()
        return loaded.model, loaded.device, int(loaded.model.config.vocab_size), c.BENCHMARK

    def _speculative_setup(self) -> SpeculativeBenchmark:
        """Load target and draft, tokenise validation prompts, and build the benchmark."""
        if self._smoke:
            target = self._tiny_model()[0]
            draft = self._tiny_model(seed=c.SEED + 1, layers=1)[0]
            generator = torch.Generator().manual_seed(c.SEED)
            prompts = [torch.randint(0, 256, (24,), generator=generator).tolist() for _ in range(2)]
            tiny = replace(
                c.BENCHMARK,
                speculative_gammas=(2,),
                speculative_prompts=2,
                warmup_runs=1,
                timed_runs=1,
            )
            return SpeculativeBenchmark(
                target, draft, torch.device("cpu"), prompts, tiny, max_new_tokens=8
            )

        loaded = ModelLoader(self._model_name, adapter=self._adapter, merge_adapter=True).load()
        draft_loaded = ModelLoader(c.DRAFT_MODEL_NAME, device=loaded.device).load()
        annotations = AnnotationLoader().load()
        manager = SplitManager()
        manifest = manager.load() if c.SPLIT_MANIFEST_PATH.exists() else manager.build(annotations)
        items = manager.assign(annotations, manifest)["val"][: self._prompts]
        builder = PromptBuilder()
        encoder = PromptEncoder(loaded.tokenizer)
        prompts = [encoder.prompt_ids(builder.build(annotation)) for annotation in items]
        label = f"{self._target_tag}:{self._model_name}"
        return SpeculativeBenchmark(
            loaded.model,
            draft_loaded.model,
            loaded.device,
            prompts,
            c.BENCHMARK,
            self._max_new_tokens,
            tokenizers=(loaded.tokenizer, draft_loaded.tokenizer),
            target_label=label,
            draft_label=c.DRAFT_MODEL_NAME,
        )

    @staticmethod
    def _tiny_model(
        seed: int = c.SEED, layers: int = 2
    ) -> tuple[PreTrainedModel, torch.device, int, c.BenchmarkConfig]:
        """Build a random model and a tiny grid for the local smoke check."""
        torch.manual_seed(seed)
        config = Qwen2Config(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
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
    parser.add_argument(
        "--model", choices=[c.MODEL_TAG_BASE, c.MODEL_TAG_FINETUNED], default="base"
    )
    parser.add_argument(
        "--adapter", type=Path, default=None, help="adapter dir for --model finetuned"
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=c.BENCHMARK.speculative_prompts,
        help="validation prompts used by the speculative suite",
    )
    parser.add_argument("--max-new-tokens", type=int, default=c.GENERATION.max_new_tokens)
    parser.add_argument("--smoke", action="store_true", help="tiny random model, tiny grid")
    args = parser.parse_args()
    adapter = args.adapter
    if args.model == c.MODEL_TAG_FINETUNED and adapter is None:
        adapter = c.ADAPTER_DIR if c.ADAPTER_DIR.exists() else c.ADAPTER_REPO
    path = Benchmarker(
        c.SMOKE_MODEL_NAME if args.smoke else c.BASE_MODEL_NAME,
        smoke=args.smoke,
        target_tag=args.model,
        adapter=adapter,
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
    ).run(args.suite)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
