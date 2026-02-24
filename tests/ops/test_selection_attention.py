# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Test suite for standalone selection attention op.

TDD: Tests written FIRST before CuTile kernel implementation.
Reference oracle: selection_attention_ref from nsa_reference.py.
"""

import math

import pytest
import torch

from tilegym.ops.nsa_reference import selection_attention_ref


def _make_block_indices(B, G, S, block_size, block_count, device="cuda"):
    """Generate valid causal block indices for testing."""
    Tc = (S + block_size - 1) // block_size
    indices = torch.zeros(B, G, S, block_count, dtype=torch.long, device=device)

    for b in range(B):
        for g in range(G):
            for s in range(S):
                nc = (s + 1) // block_size  # number of valid blocks
                if nc == 0:
                    continue
                # Pick random valid blocks (with replacement if nc < block_count)
                valid_blocks = torch.arange(nc, device=device)
                if nc >= block_count:
                    perm = torch.randperm(nc, device=device)[:block_count]
                    indices[b, g, s] = valid_blocks[perm]
                else:
                    # Fill with valid blocks, repeating if needed
                    for k in range(block_count):
                        indices[b, g, s, k] = valid_blocks[k % nc]
    return indices


def _compare_outputs(actual, expected, atol=5e-2, rtol=1e-2):
    """Compare outputs with tolerance, skipping positions with no valid blocks."""
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


# ============================================================================
# Phase 1: Simple shapes — basic correctness
# ============================================================================


class TestSelectionAttentionSimple:
    """Simple shapes for initial development and debugging."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 4, 2, 128, 64),     # baseline
            (1, 4, 2, 256, 64),     # longer S
        ],
    )
    @pytest.mark.parametrize("block_size", [32])
    @pytest.mark.parametrize("block_count", [4])
    def test_simple_shapes(self, B, HQ, G, S, D, block_size, block_count):
        """Test selection attention dispatch vs reference on simple shapes."""
        from tilegym.ops.cutile.selection_attention import selection_attention

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.bfloat16)
        block_indices = _make_block_indices(B, G, S, block_size, block_count)

        ref = selection_attention_ref(q, k, v, block_indices, block_size)
        out = selection_attention(q, k, v, block_indices, block_size)

        assert out.shape == (B, HQ, S, D)
        assert out.dtype == q.dtype
        _compare_outputs(out, ref)


# ============================================================================
# Phase 2: Regular shapes
# ============================================================================


class TestSelectionAttentionRegular:
    """Regular shapes with more heads and larger sequences."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 8, 4, 128, 128),    # more heads, larger D
            (2, 4, 2, 256, 64),     # B > 1
            (1, 16, 1, 256, 64),    # large GQA (16:1)
        ],
    )
    @pytest.mark.parametrize("block_size", [32, 64])
    @pytest.mark.parametrize("block_count", [4])
    def test_regular_shapes(self, B, HQ, G, S, D, block_size, block_count):
        """Test selection attention on regular shapes with various configs."""
        from tilegym.ops.cutile.selection_attention import selection_attention

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.bfloat16)
        block_indices = _make_block_indices(B, G, S, block_size, block_count)

        ref = selection_attention_ref(q, k, v, block_indices, block_size)
        out = selection_attention(q, k, v, block_indices, block_size)

        assert out.shape == (B, HQ, S, D)
        _compare_outputs(out, ref)


# ============================================================================
# Phase 3: Edge cases
# ============================================================================


class TestSelectionAttentionEdgeCases:
    """Irregular shapes and edge cases for full coverage."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 64, 1, 128, 64),    # large GQA (64:1)
            (1, 4, 2, 96, 64),      # irregular S
        ],
    )
    @pytest.mark.parametrize("block_size", [32])
    @pytest.mark.parametrize("block_count", [4, 8])
    def test_edge_cases(self, B, HQ, G, S, D, block_size, block_count):
        """Test selection attention on edge case shapes."""
        from tilegym.ops.cutile.selection_attention import selection_attention

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.bfloat16)
        block_indices = _make_block_indices(B, G, S, block_size, block_count)

        ref = selection_attention_ref(q, k, v, block_indices, block_size)
        out = selection_attention(q, k, v, block_indices, block_size)

        assert out.shape == (B, HQ, S, D)
        _compare_outputs(out, ref)


# ============================================================================
# Dtype tests
# ============================================================================


class TestSelectionAttentionDtype:
    """Test both float16 and bfloat16."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_dtypes(self, dtype):
        """Test selection attention with different dtypes."""
        from tilegym.ops.cutile.selection_attention import selection_attention

        torch.manual_seed(42)
        B, HQ, G, S, D = 1, 4, 2, 128, 64
        block_size, block_count = 32, 4

        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        block_indices = _make_block_indices(B, G, S, block_size, block_count)

        ref = selection_attention_ref(q, k, v, block_indices, block_size)
        out = selection_attention(q, k, v, block_indices, block_size)

        assert out.shape == (B, HQ, S, D)
        assert out.dtype == dtype
        _compare_outputs(out, ref)


# ============================================================================
# Dispatch test
# ============================================================================


class TestSelectionAttentionDispatch:
    """Test that the dispatch interface works correctly."""

    def test_dispatch_registered(self):
        """Verify selection_attention is registered as a dispatch op."""
        from tilegym.ops.ops import selection_attention

        torch.manual_seed(42)
        B, HQ, G, S, D = 1, 4, 2, 128, 64
        block_size, block_count = 32, 4

        q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.bfloat16)
        block_indices = _make_block_indices(B, G, S, block_size, block_count)

        out = selection_attention(q, k, v, block_indices, block_size, block_count)

        assert out.shape == (B, HQ, S, D)
        assert out.dtype == torch.bfloat16
