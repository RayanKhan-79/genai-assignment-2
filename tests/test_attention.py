"""Tests for reference and tiled attention (exactness, shapes, dtypes, masked rows)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from trialmatch import constants as c
from trialmatch.attention import ReferenceAttention, TiledAttention

BLOCK = 16


def tensors(
    length_q: int, length_k: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, ...]:
    """Return deterministic random Q, K, V with shape (2, 3, T/S, 8)."""
    generator = torch.Generator().manual_seed(0)
    shape_q = (2, 3, length_q, 8)
    shape_k = (2, 3, length_k, 8)
    query = torch.randn(shape_q, generator=generator).to(dtype)
    key = torch.randn(shape_k, generator=generator).to(dtype)
    value = torch.randn(shape_k, generator=generator).to(dtype)
    return query, key, value


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("length", [1, 8, 37])
def test_tiled_matches_reference_and_sdpa(causal: bool, length: int) -> None:
    """Tiled attention equals the reference and PyTorch SDPA in fp32 (T not a block multiple)."""
    query, key, value = tensors(length, length)
    tiled = TiledAttention(BLOCK, BLOCK).forward(query, key, value, causal=causal)
    reference = ReferenceAttention.forward(query, key, value, causal=causal)
    sdpa = functional.scaled_dot_product_attention(query, key, value, is_causal=causal)
    assert tiled.shape == query.shape
    torch.testing.assert_close(tiled, reference, atol=c.FP32_ATOL, rtol=0)
    torch.testing.assert_close(tiled, sdpa, atol=c.FP32_ATOL, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_tiled_matches_reference_fp16(causal: bool) -> None:
    """Half precision stays within the looser fp16 tolerance."""
    query, key, value = tensors(37, 37, torch.float16)
    tiled = TiledAttention(BLOCK, BLOCK).forward(query, key, value, causal=causal)
    reference = ReferenceAttention.forward(query, key, value, causal=causal)
    torch.testing.assert_close(tiled, reference, atol=c.FP16_ATOL, rtol=0)


def test_non_square_causal() -> None:
    """Query and key lengths may differ; the triangular mask follows positions."""
    query, key, value = tensors(5, 9)
    tiled = TiledAttention(4, 4).forward(query, key, value, causal=True)
    reference = ReferenceAttention.forward(query, key, value, causal=True)
    torch.testing.assert_close(tiled, reference, atol=c.FP32_ATOL, rtol=0)


def test_zero_query_rows_are_finite() -> None:
    """A zero query yields a uniform attention row, never NaN."""
    query, key, value = tensors(9, 9)
    query = torch.zeros_like(query)
    tiled = TiledAttention(4, 4).forward(query, key, value)
    assert torch.isfinite(tiled).all()
    torch.testing.assert_close(
        tiled, value.mean(dim=2, keepdim=True).expand_as(tiled), atol=1e-6, rtol=0
    )


def test_invalid_block_size() -> None:
    """Tile sizes must be positive."""
    with pytest.raises(ValueError):
        TiledAttention(0, 8)
