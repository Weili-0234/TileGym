# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Test suite for standalone sliding window attention op.

TDD: Tests written FIRST before kernel extraction from nsa.py.
Reference oracle: sliding_window_attention_ref from nsa_reference.py.
"""

import math

import pytest
import torch

from tilegym.ops.nsa_reference import sliding_window_attention_ref


# ============================================================================
# Phase 1: Simple shapes (develop & debug)
# ============================================================================


class TestSlidingWindowAttentionSimple:
    """Simple shapes for initial development and debugging."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 4, 2, 128, 64),   # baseline
            (1, 4, 2, 256, 64),   # longer S
        ],
    )
    @pytest.mark.parametrize("window_size", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_simple_shapes(self, B, HQ, G, S, D, window_size, dtype):
        """Test sliding window attention dispatch vs reference on simple shapes."""
        from tilegym.ops.cutile.sliding_window_attention import sliding_window_attention

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        scale = D ** -0.5

        # Reference (float32)
        o_ref = sliding_window_attention_ref(
            q.float(), k.float(), v.float(), window_size, scale
        )

        # CuTile kernel via standalone module
        o_ct = sliding_window_attention(q, k, v, window_size, scale)

        torch.testing.assert_close(
            o_ct.float(), o_ref.float(), atol=5e-2, rtol=1e-2,
            msg=f"sliding_window simple shape mismatch (dtype={dtype}, S={S}, W={window_size})",
        )


# ============================================================================
# Phase 2: Regular shapes (expand after simple pass)
# ============================================================================


class TestSlidingWindowAttentionRegular:
    """Regular shapes with more heads and larger dimensions."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 8, 4, 256, 128),   # more heads, larger D
            (1, 16, 1, 512, 64),   # large GQA (16:1)
            (2, 4, 2, 128, 64),    # B > 1
        ],
    )
    @pytest.mark.parametrize("window_size", [64, 128, 512])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_regular_shapes(self, B, HQ, G, S, D, window_size, dtype):
        """Test sliding window attention on regular shapes with various configs."""
        from tilegym.ops.cutile.sliding_window_attention import sliding_window_attention

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        scale = D ** -0.5

        o_ref = sliding_window_attention_ref(
            q.float(), k.float(), v.float(), window_size, scale
        )
        o_ct = sliding_window_attention(q, k, v, window_size, scale)

        torch.testing.assert_close(
            o_ct.float(), o_ref.float(), atol=5e-2, rtol=1e-2,
            msg=f"sliding_window regular shape mismatch (dtype={dtype})",
        )


# ============================================================================
# Phase 3: Irregular & edge cases
# ============================================================================


class TestSlidingWindowAttentionEdgeCases:
    """Irregular shapes and edge cases for full coverage."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 4, 2, 96, 64),     # S not divisible by typical block_size
            (1, 1, 1, 64, 64),     # minimal S
            (1, 64, 1, 256, 64),   # large GQA (64:1)
            (2, 4, 2, 160, 128),   # B>1, irregular S
        ],
    )
    @pytest.mark.parametrize("window_size", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_edge_cases(self, B, HQ, G, S, D, window_size, dtype):
        """Test sliding window attention on irregular and edge case shapes."""
        from tilegym.ops.cutile.sliding_window_attention import sliding_window_attention

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        scale = D ** -0.5

        o_ref = sliding_window_attention_ref(
            q.float(), k.float(), v.float(), window_size, scale
        )
        o_ct = sliding_window_attention(q, k, v, window_size, scale)

        torch.testing.assert_close(
            o_ct.float(), o_ref.float(), atol=5e-2, rtol=1e-2,
            msg=f"sliding_window edge case mismatch (S={S}, W={window_size})",
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_window_larger_than_seq(self, dtype):
        """Test window_size > S (degrades to full causal attention)."""
        from tilegym.ops.cutile.sliding_window_attention import sliding_window_attention

        torch.manual_seed(42)
        B, HQ, G, S, D = 1, 4, 2, 128, 64
        window_size = 256  # larger than S

        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        scale = D ** -0.5

        o_ref = sliding_window_attention_ref(
            q.float(), k.float(), v.float(), window_size, scale
        )
        o_ct = sliding_window_attention(q, k, v, window_size, scale)

        torch.testing.assert_close(
            o_ct.float(), o_ref.float(), atol=5e-2, rtol=1e-2,
            msg="sliding_window with W > S should degrade to full causal",
        )


# ============================================================================
# Dispatch tests: verify the op is registered and callable via dispatch
# ============================================================================


class TestSlidingWindowAttentionDispatch:
    """Test that the dispatch interface works correctly."""

    def test_dispatch_registered(self):
        """Verify sliding_window_attention is registered as a dispatch op."""
        from tilegym.ops.ops import sliding_window_attention

        torch.manual_seed(42)
        B, HQ, G, S, D = 1, 4, 2, 128, 64
        window_size = 64

        q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.bfloat16)

        o = sliding_window_attention(q, k, v, window_size=window_size)
        assert o.shape == (B, HQ, S, D)
        assert not torch.isnan(o).any()
