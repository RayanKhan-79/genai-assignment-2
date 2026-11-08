"""Greedy speculative decoding with a small draft model.

The decoder is *lossless* with respect to greedy target decoding: the emitted token sequence is
exactly the one the verifier (target) model would have produced alone. Each round drafts ``gamma``
tokens with the small model, lets the target check all of them in **one** forward pass, keeps the
longest matching prefix, and appends the target's own token at the first disagreement.

Cache bookkeeping is the only subtle part. At the start of a round both caches hold the same
prefix — the prompt plus every emitted token except the *pending* token that is about to be fed.
The draft then consumes ``pending, p1, ..., p_{gamma-1}`` (``gamma`` forward calls, producing
``p1..p_gamma``), while the target verifies ``pending, p1, ..., p_gamma`` in one call. If ``m``
proposals matched, the target must drop ``gamma - m`` positions; the draft never consumed the last
proposal, so it drops one fewer — and when every proposal matched it must consume ``p_gamma`` to
stay level with the target. Both caches then hold ``prompt + pending + p1..p_m`` again.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers import Cache, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from trialmatch import constants as c
from trialmatch.decoding import Decoder, GenerationResult, Timing


@dataclass(frozen=True)
class SpeculativeStats:
    """Acceptance statistics of one speculative run.

    Attributes:
        drafted: Tokens proposed by the draft model.
        accepted: Proposals the target model agreed with.
        rounds: Verification rounds (one target forward pass each).
        draft_forward_calls: Forward passes spent drafting.
    """

    drafted: int
    accepted: int
    rounds: int
    draft_forward_calls: int

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens the target accepted."""
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def mean_accepted_per_round(self) -> float:
        """Average number of accepted draft tokens per verification round."""
        return self.accepted / self.rounds if self.rounds else 0.0


class SpeculativeDecoder(Decoder):
    """Draft-then-verify greedy decoder built on the hand-written cache loop."""

    def __init__(
        self,
        target: PreTrainedModel,
        draft: PreTrainedModel,
        eos_token_ids: Collection[int],
        device: torch.device,
        gamma: int = c.BENCHMARK.speculative_gammas[0],
        tokenizers: tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase] | None = None,
    ) -> None:
        """Store both models and assert the tokenizers agree when they are supplied.

        Args:
            target: Verifier model (the one whose greedy output is reproduced).
            draft: Smaller proposal model.
            eos_token_ids: Tokens that stop generation.
            device: Device holding both models.
            gamma: Number of tokens drafted per round.
            tokenizers: ``(target, draft)`` tokenizers; required for a real run.

        Raises:
            ValueError: If ``gamma`` is not positive or the tokenizers differ.
        """
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        super().__init__(target, eos_token_ids, device)
        self._target = target
        self._draft = draft
        self._gamma = gamma
        if tokenizers is not None:
            self.assert_tokenizers_match(*tokenizers)

    @property
    def name(self) -> str:
        """Decoder identifier (includes gamma, so runs never collide)."""
        return f"speculative{self._gamma}"

    @staticmethod
    def assert_tokenizers_match(
        target: PreTrainedTokenizerBase, draft: PreTrainedTokenizerBase
    ) -> None:
        """Assert both tokenizers map every string to the same IDs.

        The draft may have extra *padded* embedding rows, but the vocabulary and the special tokens
        must be identical, otherwise a drafted ID would mean something else to the target.

        Raises:
            ValueError: If the vocabularies or special tokens differ.
        """
        if target.get_vocab() != draft.get_vocab():
            raise ValueError("draft and target tokenizers have different vocabularies")
        if target.all_special_tokens != draft.all_special_tokens:
            raise ValueError("draft and target tokenizers have different special tokens")

    def generate(self, input_ids: Sequence[int], max_new_tokens: int) -> GenerationResult:
        """Decode greedily, producing exactly the verifier's tokens."""
        prompt = torch.tensor([list(input_ids)], device=self._device)
        Timing.reset_peak_memory()
        Timing.sync()
        start = time.perf_counter()
        target_cache, target_logits = self._prefill(self._target, prompt)
        draft_cache, _ = self._prefill(self._draft, prompt)
        Timing.sync()
        prefill_s = time.perf_counter() - start

        pending = int(target_logits[0, -1].argmax(dim=-1).item())
        generated: list[int] = [pending]
        drafted = accepted = rounds = draft_calls = 0
        target_calls = 1

        Timing.sync()
        decode_start = time.perf_counter()
        while len(generated) < max_new_tokens and pending not in self._eos:
            rounds += 1
            budget = min(self._gamma, max_new_tokens - len(generated))
            proposals: list[int] = []
            token = pending
            for _ in range(budget):
                draft_cache, draft_logits = self._step(self._draft, draft_cache, token)
                draft_calls += 1
                token = int(draft_logits.argmax(dim=-1).item())
                proposals.append(token)

            # One verifier forward over [pending] + proposals: logits[i] follows sequence i.
            target_cache, verifier = self._verify(self._target, target_cache, [pending, *proposals])
            target_calls += 1
            predictions = verifier.argmax(dim=-1).tolist()

            matched = 0
            while matched < budget and proposals[matched] == predictions[matched]:
                matched += 1
            drafted += budget
            accepted += matched

            target_cache.crop(matched - budget)
            draft_drop = budget - matched - 1
            if draft_drop > 0:
                draft_cache.crop(-draft_drop)
            elif draft_drop < 0:
                # Every proposal matched: the draft must consume the last one it has not seen.
                draft_cache, _ = self._step(self._draft, draft_cache, proposals[-1])
                draft_calls += 1

            # Accepted drafts plus the target's own token at the first disagreement.
            for token_id in [*proposals[:matched], int(predictions[matched])]:
                if len(generated) >= max_new_tokens:
                    break
                generated.append(token_id)
                if token_id in self._eos:
                    break
            pending = generated[-1]
        Timing.sync()
        decode_s = time.perf_counter() - decode_start

        stats = SpeculativeStats(
            drafted=drafted,
            accepted=accepted,
            rounds=rounds,
            draft_forward_calls=draft_calls,
        )
        return GenerationResult(
            tokens=tuple(generated),
            prefill_s=prefill_s,
            decode_s=decode_s,
            peak_mem_bytes=Timing.peak_memory_bytes(),
            n_forward_calls=target_calls,
            stats=stats,
        )

    def _prefill(self, model: PreTrainedModel, prompt: torch.Tensor) -> tuple[Cache, torch.Tensor]:
        """Run one forward over the prompt and return ``(cache, logits)``."""
        length = prompt.shape[1]
        output = self._call(
            model,
            input_ids=prompt,
            attention_mask=torch.ones((1, length), dtype=torch.long, device=self._device),
            position_ids=torch.arange(length, device=self._device).unsqueeze(0),
            use_cache=True,
        )
        return self._unwrap(output)

    def _step(self, model: PreTrainedModel, cache: Cache, token: int) -> tuple[Cache, torch.Tensor]:
        """Feed one token to ``model`` with its cache and return ``(cache, last logits)``."""
        length = cache.get_seq_length()
        output = self._call(
            model,
            input_ids=torch.tensor([[token]], device=self._device),
            attention_mask=torch.ones((1, length + 1), dtype=torch.long, device=self._device),
            position_ids=torch.tensor([[length]], device=self._device),
            past_key_values=cache,
            use_cache=True,
        )
        new_cache, logits = self._unwrap(output)
        return new_cache, logits[0, -1]

    def _verify(
        self, model: PreTrainedModel, cache: Cache, tokens: Sequence[int]
    ) -> tuple[Cache, torch.Tensor]:
        """Run the verifier over ``tokens`` with the cache and return ``(cache, logits)``."""
        length = cache.get_seq_length()
        count = len(tokens)
        output = self._call(
            model,
            input_ids=torch.tensor([list(tokens)], device=self._device),
            attention_mask=torch.ones((1, length + count), dtype=torch.long, device=self._device),
            position_ids=torch.arange(length, length + count, device=self._device).unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        )
        new_cache, logits = self._unwrap(output)
        return new_cache, logits[0]

    @staticmethod
    def _call(model: PreTrainedModel, **kwargs: Any) -> CausalLMOutputWithPast:
        """Forward one model with a typed output."""
        return cast(CausalLMOutputWithPast, model(**kwargs))

    @staticmethod
    def _unwrap(output: CausalLMOutputWithPast) -> tuple[Cache, torch.Tensor]:
        """Return the cache and logits of a forward pass, rejecting a missing cache."""
        if output.past_key_values is None or output.logits is None:
            raise ValueError("model did not return a KV cache and logits")
        return output.past_key_values, output.logits
