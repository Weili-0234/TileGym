# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Test suite for standalone top-K block selection op.

TDD: Tests written FIRST before CuTile kernel implementation.
Reference oracle: topk_from_importance_ref from nsa_reference.py.
"""

import math

import pytest
import torch

from tilegym.ops.nsa_reference import topk_from_importance_ref


def _make_random_importance(B, G, S, block_size, device="cuda"):
    """Generate random importance scores with proper causal masking."""
    Tc = (S + block_size - 1) // block_size
    importance = torch.randn(B, G, S, Tc, device=device, dtype=torch.float32)
    return importance


# ============================================================================
# Phase 1: Simple shapes — basic correctness
# ============================================================================


class TestTopKBlockSelectionSimple:
    """Simple shapes for initial development and debugging."""

    @pytest.mark.parametrize(
        "B, G, S, block_size",
        [
            (1, 2, 128, 32),   # S=128, BS=32, Tc=4
            (1, 2, 256, 32),   # S=256, BS=32, Tc=8
        ],
    )
    @pytest.mark.parametrize("block_count", [4])
    @pytest.mark.parametrize("num_init, num_local", [(0, 0), (1, 2)])
    def test_simple_shapes(self, B, G, S, block_size, block_count, num_init, num_local):
        """Test top-K block selection dispatch vs reference on simple shapes."""
        from tilegym.ops.cutile.topk_block_selection import topk_block_selection

        torch.manual_seed(42)
        importance = _make_random_importance(B, G, S, block_size)

        # Reference
        indices_ref = topk_from_importance_ref(
            importance, block_size, block_count, num_init, num_local
        )

        # CuTile kernel
        indices_ct = topk_block_selection(
            importance, block_size, block_count, num_init, num_local
        )

        assert indices_ct.shape == (B, G, S, block_count)

        # Compare as SETS per position (order may differ)
        for b in range(B):
            for g in range(G):
                for s in range(S):
                    nc = (s + 1) // block_size
                    if nc == 0:
                        continue  # no valid blocks
                    actual_k = min(block_count, nc)
                    ref_set = set(indices_ref[b, g, s, :actual_k].tolist())
                    ct_set = set(indices_ct[b, g, s, :actual_k].tolist())
                    assert ref_set == ct_set, (
                        f"TopK mismatch at (b={b}, g={g}, s={s}): "
                        f"ref={ref_set}, ct={ct_set}"
                    )


# ============================================================================
# Phase 2: Regular shapes
# ============================================================================


class TestTopKBlockSelectionRegular:
    """Regular shapes with more heads and larger sequences."""

    @pytest.mark.parametrize(
        "B, G, S, block_size",
        [
            (1, 4, 512, 32),   # larger Tc=16
            (2, 2, 256, 32),   # B > 1
            (1, 2, 256, 64),   # larger block_size
        ],
    )
    @pytest.mark.parametrize("block_count", [4, 8])
    @pytest.mark.parametrize("num_init, num_local", [(0, 0), (1, 2)])
    def test_regular_shapes(self, B, G, S, block_size, block_count, num_init, num_local):
        """Test top-K on regular shapes with various configs."""
        from tilegym.ops.cutile.topk_block_selection import topk_block_selection

        torch.manual_seed(42)
        importance = _make_random_importance(B, G, S, block_size)

        indices_ref = topk_from_importance_ref(
            importance, block_size, block_count, num_init, num_local
        )
        indices_ct = topk_block_selection(
            importance, block_size, block_count, num_init, num_local
        )

        assert indices_ct.shape == (B, G, S, block_count)

        # Compare as sets per position
        for b in range(B):
            for g in range(G):
                for s in range(S):
                    nc = (s + 1) // block_size
                    if nc == 0:
                        continue
                    actual_k = min(block_count, nc)
                    ref_set = set(indices_ref[b, g, s, :actual_k].tolist())
                    ct_set = set(indices_ct[b, g, s, :actual_k].tolist())
                    assert ref_set == ct_set, (
                        f"TopK mismatch at (b={b}, g={g}, s={s}): "
                        f"ref={ref_set}, ct={ct_set}"
                    )


# ============================================================================
# Phase 3: Edge cases
# ============================================================================


class TestTopKBlockSelectionEdgeCases:
    """Irregular shapes and edge cases for full coverage."""

    @pytest.mark.parametrize(
        "B, G, S, block_size",
        [
            (1, 1, 64, 32),    # Tc=2 (only 2 blocks)
            (1, 2, 96, 32),    # irregular S (96/32=3)
        ],
    )
    @pytest.mark.parametrize("block_count", [4, 8])
    @pytest.mark.parametrize("num_init, num_local", [(0, 0)])
    def test_edge_cases(self, B, G, S, block_size, block_count, num_init, num_local):
        """Test top-K on edge case shapes."""
        from tilegym.ops.cutile.topk_block_selection import topk_block_selection

        torch.manual_seed(42)
        importance = _make_random_importance(B, G, S, block_size)

        indices_ref = topk_from_importance_ref(
            importance, block_size, block_count, num_init, num_local
        )
        indices_ct = topk_block_selection(
            importance, block_size, block_count, num_init, num_local
        )

        assert indices_ct.shape == (B, G, S, block_count)

        for b in range(B):
            for g in range(G):
                for s in range(S):
                    nc = (s + 1) // block_size
                    if nc == 0:
                        continue
                    actual_k = min(block_count, nc)
                    ref_set = set(indices_ref[b, g, s, :actual_k].tolist())
                    ct_set = set(indices_ct[b, g, s, :actual_k].tolist())
                    assert ref_set == ct_set


# ============================================================================
# Reserved block tests
# ============================================================================


class TestTopKReservedBlocks:
    """Verify that init and local blocks are always included in output."""

    @pytest.mark.parametrize(
        "B, G, S, block_size",
        [
            (1, 2, 256, 32),
            (1, 4, 512, 64),
        ],
    )
    @pytest.mark.parametrize("block_count", [8])
    def test_init_blocks_always_included(self, B, G, S, block_size, block_count):
        """Verify first num_init blocks are always in the output."""
        from tilegym.ops.cutile.topk_block_selection import topk_block_selection

        torch.manual_seed(42)
        num_init = 2
        num_local = 0
        importance = _make_random_importance(B, G, S, block_size)

        indices = topk_block_selection(
            importance, block_size, block_count, num_init, num_local
        )

        for b in range(B):
            for g in range(G):
                for s in range(S):
                    nc = (s + 1) // block_size
                    if nc == 0:
                        continue
                    idx_set = set(indices[b, g, s].tolist())
                    # First num_init blocks should be in output (if they're valid)
                    for i in range(min(num_init, nc)):
                        assert i in idx_set, (
                            f"Init block {i} missing at (b={b}, g={g}, s={s}), "
                            f"got {idx_set}"
                        )

    @pytest.mark.parametrize(
        "B, G, S, block_size",
        [
            (1, 2, 256, 32),
            (1, 4, 512, 64),
        ],
    )
    @pytest.mark.parametrize("block_count", [8])
    def test_local_blocks_always_included(self, B, G, S, block_size, block_count):
        """Verify nearest num_local blocks are always in the output."""
        from tilegym.ops.cutile.topk_block_selection import topk_block_selection

        torch.manual_seed(42)
        num_init = 0
        num_local = 2
        importance = _make_random_importance(B, G, S, block_size)

        indices = topk_block_selection(
            importance, block_size, block_count, num_init, num_local
        )

        for b in range(B):
            for g in range(G):
                for s in range(S):
                    nc = (s + 1) // block_size
                    if nc == 0:
                        continue
                    idx_set = set(indices[b, g, s].tolist())
                    current_block = s // block_size
                    for i in range(num_local):
                        local_idx = current_block - i
                        if 0 <= local_idx < nc:
                            assert local_idx in idx_set, (
                                f"Local block {local_idx} missing at (b={b}, g={g}, s={s}), "
                                f"got {idx_set}"
                            )


# ============================================================================
# Dispatch test
# ============================================================================


class TestTopKBlockSelectionDispatch:
    """Test that the dispatch interface works correctly."""

    def test_dispatch_registered(self):
        """Verify topk_block_selection is registered as a dispatch op."""
        from tilegym.ops.ops import topk_block_selection

        torch.manual_seed(42)
        B, G, S = 1, 2, 128
        block_size = 32
        block_count = 4
        Tc = S // block_size

        importance = torch.randn(B, G, S, Tc, device="cuda", dtype=torch.float32)
        indices = topk_block_selection(importance, block_size, block_count)

        assert indices.shape == (B, G, S, block_count)
        assert indices.dtype == torch.long
