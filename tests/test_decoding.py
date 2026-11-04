"""Tests for the hand-written greedy decoders on a tiny random Qwen2 model (CPU, fp32)."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from trialmatch.decoding import CachedDecoder, DecoderFactory, UncachedDecoder

VOCAB = 128


def tiny_model(seed: int = 0) -> Qwen2ForCausalLM:
    """Build a 2-layer random Qwen2 model small enough for fast CPU tests."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    # transformers' config-driven constructors are not annotated for mypy.
    model = Qwen2ForCausalLM(config)  # type: ignore[no-untyped-call]
    return cast(Qwen2ForCausalLM, cast(Any, model).eval())


def prompt(length: int, seed: int = 1) -> list[int]:
    """Return a deterministic random prompt of ``length`` token IDs."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (length,), generator=generator).tolist()


@pytest.mark.parametrize("prompt_len", [5, 17, 40])
@pytest.mark.parametrize("new_tokens", [1, 8])
def test_cached_matches_uncached(prompt_len: int, new_tokens: int) -> None:
    """Greedy decoding is token-identical with and without the KV cache (fp32)."""
    model = tiny_model()
    device = torch.device("cpu")
    uncached = UncachedDecoder(model, eos_token_ids=[], device=device)
    cached = CachedDecoder(model, eos_token_ids=[], device=device)

    reference = uncached.generate(prompt(prompt_len), new_tokens)
    result = cached.generate(prompt(prompt_len), new_tokens)

    assert result.tokens == reference.tokens
    assert len(result.tokens) == new_tokens
    assert result.n_forward_calls == new_tokens
    assert reference.n_forward_calls == new_tokens


def test_cached_is_deterministic() -> None:
    """Two cached runs on the same prompt return the same tokens and stable timing fields."""
    decoder = CachedDecoder(tiny_model(), eos_token_ids=[], device=torch.device("cpu"))
    first = decoder.generate(prompt(12), 6)
    second = decoder.generate(prompt(12), 6)
    assert first.tokens == second.tokens
    assert first.n_forward_calls == second.n_forward_calls == 6


def test_eos_stops_generation() -> None:
    """The decoder stops as soon as EOS is produced and keeps it in the output."""
    model = tiny_model()
    device = torch.device("cpu")
    probe = CachedDecoder(model, eos_token_ids=[], device=device).generate(prompt(9), 1)
    eos = probe.tokens[0]

    result = CachedDecoder(model, eos_token_ids=[eos], device=device).generate(prompt(9), 8)
    assert result.tokens == (eos,)
    assert result.n_forward_calls == 1


def test_timing_fields_are_consistent() -> None:
    """The prefill/decode split adds up to the reported latency and is never negative."""
    result = CachedDecoder(tiny_model(), eos_token_ids=[], device=torch.device("cpu")).generate(
        prompt(11), 5
    )
    assert result.prefill_s >= 0.0
    assert result.decode_s >= 0.0
    assert result.latency_s == pytest.approx(result.prefill_s + result.decode_s)
    assert result.peak_mem_bytes == 0  # CPU has no CUDA allocator to measure


def test_factory_and_unknown_kind() -> None:
    """The factory returns the requested class and rejects unknown names."""
    model = tiny_model()
    device = torch.device("cpu")
    assert isinstance(DecoderFactory.create("cached", model, [], device), CachedDecoder)
    assert isinstance(DecoderFactory.create("uncached", model, [], device), UncachedDecoder)
    with pytest.raises(ValueError):
        DecoderFactory.create("beam", model, [], device)
