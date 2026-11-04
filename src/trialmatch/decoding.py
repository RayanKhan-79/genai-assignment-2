"""Hand-written greedy decoders: an uncached reference and an explicit KV-cache loop.

Both decoders produce the same token sequence for the same prompt (greedy, no sampling) and differ
only in how much work each step repeats. Timings are wall-clock around synchronised regions.

Why the cached path is correct
------------------------------
* **Positions.** Qwen applies RoPE with an *absolute* index, so the incremental step must pass
  ``position_ids = [[cache_len]]`` instead of letting the model start again at 0.
* **Masking.** One query attends to every cached key plus itself, so the attention mask has length
  ``cache_len + 1`` while the query length is 1. The model derives the causal mask from
  ``past_key_values`` and that attention mask.
* **Cache growth.** Per layer the cache holds ``2 x n_kv_heads x head_dim`` values per position, so
  with GQA (12 query heads, 2 KV heads) it is six times smaller than a multi-head cache.
* **Numerical differences.** The cached and uncached paths batch different matmul shapes and
  reduction orders, so fp16 logits can differ in the last bits; a near-tie can flip the argmax.
  The tests assert exact equality in fp32 on a tiny model and the benchmark reports real-model
  mismatches instead of hiding them.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import Cache, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

if TYPE_CHECKING:
    from trialmatch.speculative import SpeculativeStats


@dataclass(frozen=True)
class GenerationResult:
    """Tokens produced by one decoding run plus its cost breakdown.

    Attributes:
        tokens: Newly generated token IDs (EOS included when it was produced).
        prefill_s: Seconds spent in the first forward pass over the prompt.
        decode_s: Seconds spent in all subsequent forward passes.
        peak_mem_bytes: CUDA peak allocated memory during the run (0 off CUDA).
        n_forward_calls: Number of model forward passes.
        stats: Speculative-decoding acceptance statistics, when applicable.
    """

    tokens: tuple[int, ...]
    prefill_s: float
    decode_s: float
    peak_mem_bytes: int
    n_forward_calls: int
    stats: SpeculativeStats | None = None

    @property
    def latency_s(self) -> float:
        """Total generation latency in seconds."""
        return self.prefill_s + self.decode_s


class Timing:
    """Synchronised wall-clock timing and CUDA peak-memory measurement."""

    @staticmethod
    def sync() -> None:
        """Synchronise CUDA so timings include the queued kernels."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def reset_peak_memory() -> None:
        """Reset the CUDA peak-memory counter (no-op on CPU)."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def peak_memory_bytes() -> int:
        """Return the CUDA peak allocated bytes since the last reset (0 on CPU)."""
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_allocated())


class Decoder(ABC):
    """Common state for greedy decoders, plus the numeric ``generate`` contract."""

    def __init__(
        self,
        model: PreTrainedModel,
        eos_token_ids: Collection[int],
        device: torch.device,
    ) -> None:
        """Store the model, the stop tokens and the device the model lives on."""
        self._model = model
        self._eos = set(eos_token_ids)
        self._device = device

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result filenames and tables."""

    @abstractmethod
    def generate(self, input_ids: Sequence[int], max_new_tokens: int) -> GenerationResult:
        """Greedily decode at most ``max_new_tokens`` tokens after ``input_ids``."""

    def _forward(self, **kwargs: Any) -> CausalLMOutputWithPast:
        """Run one forward pass and return the typed output object."""
        return cast(CausalLMOutputWithPast, self._model(**kwargs))

    @staticmethod
    def _argmax_token(output: CausalLMOutputWithPast) -> int:
        """Return the greedy next token from the last logit position."""
        logits = output.logits
        if logits is None:
            raise ValueError("model returned no logits")
        return int(logits[0, -1].argmax(dim=-1).item())


class UncachedDecoder(Decoder):
    """Reference decoder: re-run the whole sequence every step, no cache at all."""

    @property
    def name(self) -> str:
        """Decoder identifier."""
        return "uncached"

    def generate(self, input_ids: Sequence[int], max_new_tokens: int) -> GenerationResult:
        """Decode greedily with ``use_cache=False`` on the full sequence every step."""
        tokens = torch.tensor([list(input_ids)], device=self._device)
        generated: list[int] = []
        prefill_s = 0.0
        decode_s = 0.0
        calls = 0
        Timing.reset_peak_memory()
        for step in range(max_new_tokens):
            Timing.sync()
            start = time.perf_counter()
            output = self._forward(input_ids=tokens, use_cache=False)
            Timing.sync()
            elapsed = time.perf_counter() - start
            if step == 0:
                prefill_s = elapsed
            else:
                decode_s += elapsed
            calls += 1
            next_token = self._argmax_token(output)
            generated.append(next_token)
            if next_token in self._eos or len(generated) == max_new_tokens:
                break
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=self._device)], dim=1)
        return GenerationResult(
            tokens=tuple(generated),
            prefill_s=prefill_s,
            decode_s=decode_s,
            peak_mem_bytes=Timing.peak_memory_bytes(),
            n_forward_calls=calls,
        )


class CachedDecoder(Decoder):
    """Explicit KV-cache decoder: one prefill, then one token per forward pass."""

    @property
    def name(self) -> str:
        """Decoder identifier."""
        return "cached"

    def generate(self, input_ids: Sequence[int], max_new_tokens: int) -> GenerationResult:
        """Decode greedily while growing a ``DynamicCache`` one position at a time."""
        prompt = torch.tensor([list(input_ids)], device=self._device)
        batch, prompt_len = prompt.shape
        generated: list[int] = []
        Timing.reset_peak_memory()

        attention_mask = torch.ones((batch, prompt_len), dtype=torch.long, device=self._device)
        position_ids = torch.arange(prompt_len, device=self._device).unsqueeze(0)
        Timing.sync()
        start = time.perf_counter()
        output = self._forward(
            input_ids=prompt,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        Timing.sync()
        prefill_s = time.perf_counter() - start
        cache = cast(Cache, output.past_key_values)
        calls = 1

        decode_s = 0.0
        for _ in range(max_new_tokens):
            next_token = self._argmax_token(output)
            generated.append(next_token)
            if next_token in self._eos or len(generated) == max_new_tokens:
                break
            cache_len = cache.get_seq_length()
            step_ids = torch.tensor([[next_token]], device=self._device)
            attention_mask = torch.ones(
                (batch, cache_len + 1), dtype=torch.long, device=self._device
            )
            # RoPE is absolute: the single new query sits at index ``cache_len``.
            position_ids = torch.tensor([[cache_len]], device=self._device)
            Timing.sync()
            start = time.perf_counter()
            output = self._forward(
                input_ids=step_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            Timing.sync()
            decode_s += time.perf_counter() - start
            cache = cast(Cache, output.past_key_values)
            calls += 1

        return GenerationResult(
            tokens=tuple(generated),
            prefill_s=prefill_s,
            decode_s=decode_s,
            peak_mem_bytes=Timing.peak_memory_bytes(),
            n_forward_calls=calls,
        )


class DecoderFactory:
    """Map a decoder name to a concrete :class:`Decoder`."""

    @staticmethod
    def create(
        kind: str, model: PreTrainedModel, eos_token_ids: Collection[int], device: torch.device
    ) -> Decoder:
        """Return the decoder registered under ``kind``.

        Raises:
            ValueError: If the name is unknown.
        """
        if kind == "uncached":
            return UncachedDecoder(model, eos_token_ids, device)
        if kind == "cached":
            return CachedDecoder(model, eos_token_ids, device)
        raise ValueError(f"unknown decoder {kind!r}")
