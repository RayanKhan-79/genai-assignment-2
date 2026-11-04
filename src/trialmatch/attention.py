"""Attention: a readable reference implementation and a tiled online-softmax implementation.

Shapes are ``(batch, heads, queries, head_dim)`` throughout. The reference builds the full
``T x T`` score matrix; the tiled version never materialises more than one query block by one key
block, which is the memory argument in the report. Both are pure PyTorch — no fused kernels — so the
Python loop is expected to be *slower* than ``scaled_dot_product_attention`` (see the benchmark).
"""

from __future__ import annotations

import math

import torch

from trialmatch import constants as c


class ReferenceAttention:
    """Textbook attention with a full score matrix.

    Stability comes from ``torch.softmax``, which subtracts the row maximum internally before
    exponentiating. No manual max-subtraction is performed here.
    """

    @staticmethod
    def additive_mask(length_q: int, length_k: int, device: torch.device) -> torch.Tensor:
        """Return a boolean mask that is ``True`` for positions a causal query may not see."""
        positions_q = torch.arange(length_q, device=device).unsqueeze(1)
        positions_k = torch.arange(length_k, device=device).unsqueeze(0)
        return positions_k > positions_q

    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """Compute ``softmax(QK^T / sqrt(D) + mask) V`` with a full matrix.

        Args:
            query: ``(B, H, T, D)`` queries.
            key: ``(B, H, S, D)`` keys.
            value: ``(B, H, S, D)`` values.
            causal: Whether query ``i`` may only see keys ``j <= i``.

        Returns:
            ``(B, H, T, D)`` attention output.
        """
        scale = 1.0 / math.sqrt(query.shape[-1])
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        if causal:
            mask = ReferenceAttention.additive_mask(query.shape[-2], key.shape[-2], query.device)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)


class TiledAttention:
    """FlashAttention-style tiling with an online (running) softmax.

    For each query block the loop keeps a running row maximum ``m``, a running normaliser ``l`` and
    a running accumulator ``acc``. A new key block rescales the old accumulator by
    ``exp(m_old - m_new)`` before adding its own contribution, so the result equals the full
    softmax without ever storing it. Fully masked key blocks are skipped in the causal case.
    """

    def __init__(
        self,
        block_q: int = c.BENCHMARK.attention_block_q,
        block_k: int = c.BENCHMARK.attention_block_k,
    ) -> None:
        """Store the tile sizes in queries and keys."""
        if block_q <= 0 or block_k <= 0:
            raise ValueError("block sizes must be positive")
        self._block_q = block_q
        self._block_k = block_k

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """Compute attention tile by tile; numerically equal to :class:`ReferenceAttention`.

        Args:
            query: ``(B, H, T, D)`` queries.
            key: ``(B, H, S, D)`` keys.
            value: ``(B, H, S, D)`` values.
            causal: Whether query ``i`` may only see keys ``j <= i``.

        Returns:
            ``(B, H, T, D)`` attention output; fully masked rows yield zeros (never NaN).
        """
        batch, heads, length_q, head_dim = query.shape
        length_k = key.shape[2]
        scale = 1.0 / math.sqrt(head_dim)
        output = torch.zeros_like(query)
        for start_q in range(0, length_q, self._block_q):
            end_q = min(start_q + self._block_q, length_q)
            rows = end_q - start_q
            query_block = query[:, :, start_q:end_q]
            running_max = torch.full(
                (batch, heads, rows, 1), -torch.inf, device=query.device, dtype=query.dtype
            )
            running_sum = torch.zeros(
                (batch, heads, rows, 1), device=query.device, dtype=query.dtype
            )
            accumulator = torch.zeros(
                (batch, heads, rows, head_dim), device=query.device, dtype=query.dtype
            )
            for start_k in range(0, length_k, self._block_k):
                if causal and start_k > end_q - 1:
                    break  # every key in this block lies after every query in the row block
                end_k = min(start_k + self._block_k, length_k)
                key_block = key[:, :, start_k:end_k]
                scores = torch.matmul(query_block, key_block.transpose(-1, -2)) * scale
                if causal:
                    positions_q = torch.arange(start_q, end_q, device=query.device).unsqueeze(1)
                    positions_k = torch.arange(start_k, end_k, device=query.device).unsqueeze(0)
                    # True -inf (not finfo.min) so a fully masked block stays exactly zero.
                    scores = scores.masked_fill(positions_k > positions_q, float("-inf"))
                block_max = scores.amax(dim=-1, keepdim=True)
                new_max = torch.maximum(running_max, block_max)
                # A fully masked row has max -inf; clamping to 0 makes exp(-inf - 0) == 0, so the
                # row stays zero instead of producing NaN.
                safe_max = torch.where(torch.isinf(new_max), torch.zeros_like(new_max), new_max)
                correction = torch.exp(running_max - safe_max)
                probabilities = torch.exp(scores - safe_max)
                accumulator = accumulator * correction + torch.matmul(
                    probabilities, value[:, :, start_k:end_k]
                )
                running_sum = running_sum * correction + probabilities.sum(dim=-1, keepdim=True)
                running_max = new_max
            divisor = torch.where(running_sum == 0, torch.ones_like(running_sum), running_sum)
            output[:, :, start_q:end_q] = accumulator / divisor
        return output
