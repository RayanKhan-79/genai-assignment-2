"""Tests for greedy speculative decoding on tiny random Qwen2 models (CPU, fp32).

The contract under test is *losslessness*: with any draft model the emitted tokens must equal what
the target would greedily emit alone. A second suite uses a copy of the target as the draft, where
every proposal must be accepted, which pins down the cache-cropping bookkeeping.
"""

from __future__ import annotations

import copy
from typing import Any, cast

import pytest
import torch
from transformers import PreTrainedTokenizerBase, Qwen2Config, Qwen2ForCausalLM

from trialmatch.decoding import CachedDecoder
from trialmatch.speculative import SpeculativeDecoder, SpeculativeStats

VOCAB = 128


def tiny_model(layers: int, seed: int = 0) -> Qwen2ForCausalLM:
    """Build a random Qwen2 model small enough for fast CPU tests."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
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


def reference_tokens(model: Qwen2ForCausalLM, ids: list[int], new_tokens: int) -> tuple[int, ...]:
    """Greedy tokens from the hand-written cached decoder, i.e. the target's own output."""
    return (
        CachedDecoder(model, eos_token_ids=[], device=torch.device("cpu"))
        .generate(ids, new_tokens)
        .tokens
    )


@pytest.mark.parametrize("gamma", [1, 3, 5])
@pytest.mark.parametrize("prompt_len", [5, 23])
def test_lossless_with_different_draft(gamma: int, prompt_len: int) -> None:
    """Any draft model still yields exactly the target's greedy sequence."""
    target = tiny_model(layers=2, seed=0)
    draft = tiny_model(layers=1, seed=7)
    decoder = SpeculativeDecoder(
        target, draft, eos_token_ids=[], device=torch.device("cpu"), gamma=gamma
    )
    result = decoder.generate(prompt(prompt_len), 12)
    assert result.tokens == reference_tokens(target, prompt(prompt_len), 12)


def test_identical_draft_accepts_every_proposal() -> None:
    """With the target as its own draft, every proposal matches and caches stay aligned."""
    target = tiny_model(layers=2, seed=3)
    draft = copy.deepcopy(target)
    decoder = SpeculativeDecoder(
        target, draft, eos_token_ids=[], device=torch.device("cpu"), gamma=4
    )
    result = decoder.generate(prompt(19), 13)
    stats = result.stats
    assert isinstance(stats, SpeculativeStats)
    assert result.tokens == reference_tokens(target, prompt(19), 13)
    assert len(result.tokens) == 13
    assert stats.drafted == stats.accepted
    assert stats.acceptance_rate == 1.0
    assert stats.draft_forward_calls == stats.drafted + stats.rounds
    assert stats.mean_accepted_per_round == stats.accepted / stats.rounds


def test_long_generation_with_full_acceptance() -> None:
    """Many full-acceptance rounds do not drift the caches over a long generation."""
    target = tiny_model(layers=2, seed=5)
    draft = copy.deepcopy(target)
    decoder = SpeculativeDecoder(
        target, draft, eos_token_ids=[], device=torch.device("cpu"), gamma=3
    )
    result = decoder.generate(prompt(11), 60)
    assert result.tokens == reference_tokens(target, prompt(11), 60)
    stats = result.stats
    assert isinstance(stats, SpeculativeStats)
    assert stats.accepted == stats.drafted
    assert stats.draft_forward_calls == stats.drafted + stats.rounds


class ForcedDraft(torch.nn.Module):
    """Draft model whose next token is scripted, so acceptance can be forced.

    The base forward still builds a real KV cache, which keeps the cache bookkeeping under test;
    only the logits are replaced, one scripted token per call.
    """

    def __init__(self, base: Qwen2ForCausalLM, tokens: list[int]) -> None:
        """Store the base model and the token to force at each single-token draft step."""
        super().__init__()
        if not tokens:
            raise ValueError("at least one scripted token is required")
        self._base = base
        self._tokens = tokens
        self._calls = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate the forward pass and overwrite the logits with a one-hot script."""
        output = self._base(*args, **kwargs)
        if output.logits is None:
            raise ValueError("base model returned no logits")
        # The prefill call only warms the cache; the script advances on one-token steps.
        length = int(kwargs["input_ids"].shape[1])
        index = 0
        if length == 1:
            index = min(self._calls, len(self._tokens) - 1)
            self._calls += 1
        logits = torch.full_like(output.logits, -100.0)
        logits[..., self._tokens[index]] = 100.0
        output.logits = logits
        return output


def test_rejected_proposals_are_cropped() -> None:
    """A draft that never agrees must still leave the target's exact greedy tokens.

    This is the regression test for ``Cache.crop``: transformers 5 rejects a positive count, so a
    rejected proposal used to abort the warm-up run instead of cropping the verifier cache.
    """
    target = tiny_model(layers=2, seed=0)
    ids = prompt(7)
    reference = reference_tokens(target, ids, 6)
    wrong = next(token for token in range(VOCAB) if token not in reference)
    draft = cast(Any, ForcedDraft(target, [wrong] * 32))
    decoder = SpeculativeDecoder(target, draft, [], torch.device("cpu"), gamma=3)
    result = decoder.generate(ids, 6)
    assert result.tokens == reference
    stats = result.stats
    assert isinstance(stats, SpeculativeStats)
    assert stats.accepted == 0
    assert stats.drafted > 0


def test_partial_acceptance_matches_target() -> None:
    """A mix of accepted and rejected proposals still reproduces the target's greedy sequence."""
    target = tiny_model(layers=2, seed=1)
    ids = prompt(9)
    reference = reference_tokens(target, ids, 10)
    wrong = next(token for token in range(VOCAB) if token not in reference)
    scripted: list[int] = []
    for position in range(1, 24):
        scripted.append(reference[position % len(reference)] if position % 2 else wrong)
    draft = cast(Any, ForcedDraft(target, scripted))
    decoder = SpeculativeDecoder(target, draft, [], torch.device("cpu"), gamma=2)
    result = decoder.generate(ids, 10)
    assert result.tokens == reference
    stats = result.stats
    assert isinstance(stats, SpeculativeStats)
    assert 0 < stats.accepted < stats.drafted


def test_eos_stops_generation() -> None:
    """The first emitted token being EOS ends the run immediately and is kept."""
    target = tiny_model(layers=2, seed=4)
    eos = reference_tokens(target, prompt(9), 1)[0]
    decoder = SpeculativeDecoder(
        target, tiny_model(layers=1, seed=2), [eos], torch.device("cpu"), gamma=3
    )
    result = decoder.generate(prompt(9), 25)
    assert result.tokens == (eos,)
    assert result.n_forward_calls == 1
    assert result.stats is not None
    assert result.stats.rounds == 0


def test_eos_inside_accepted_proposals() -> None:
    """An EOS matched among the proposals stops the run and is the last token."""
    target = tiny_model(layers=2, seed=6)
    draft = copy.deepcopy(target)
    ids = prompt(13)
    first = reference_tokens(target, ids, 1)[0]
    decoder = SpeculativeDecoder(target, draft, [first], torch.device("cpu"), gamma=4)
    result = decoder.generate(ids, 20)
    assert result.tokens == (first,)


def test_gamma_must_be_positive() -> None:
    """Gamma is the draft budget and must be a positive integer."""
    target = tiny_model(layers=1)
    with pytest.raises(ValueError, match="gamma"):
        SpeculativeDecoder(target, target, [], torch.device("cpu"), gamma=0)


class StubTokenizer:
    """Minimal tokenizer surface used by :meth:`assert_tokenizers_match`."""

    def __init__(self, vocab: dict[str, int], specials: list[str]) -> None:
        """Store the fake vocabulary and special-token list."""
        self._vocab = vocab
        self.all_special_tokens = specials

    def get_vocab(self) -> dict[str, int]:
        """Return the fake vocabulary."""
        return self._vocab


def test_tokenizers_must_match() -> None:
    """Differing vocabularies or special tokens are rejected before any generation."""
    a = cast(PreTrainedTokenizerBase, StubTokenizer({"x": 0}, ["<eos>"]))
    b = cast(PreTrainedTokenizerBase, StubTokenizer({"x": 0, "y": 1}, ["<eos>"]))
    c = cast(PreTrainedTokenizerBase, StubTokenizer({"x": 0}, ["<pad>"]))
    target = tiny_model(layers=1)
    with pytest.raises(ValueError, match="vocabular"):
        SpeculativeDecoder(target, target, [], torch.device("cpu"), tokenizers=(a, b))
    with pytest.raises(ValueError, match="special"):
        SpeculativeDecoder(target, target, [], torch.device("cpu"), tokenizers=(a, c))


def test_stats_properties_on_empty_run() -> None:
    """A run with no drafts reports zeros instead of dividing by zero."""
    stats = SpeculativeStats(drafted=0, accepted=0, rounds=0, draft_forward_calls=0)
    assert stats.acceptance_rate == 0.0
    assert stats.mean_accepted_per_round == 0.0
