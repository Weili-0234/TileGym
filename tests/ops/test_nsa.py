# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Test suite for NSA (Native Sparse Attention) forward pass.

Layer 0: Cross-validate reference functions against FLA naive
Layer 1: Unit tests for each cutile kernel vs reference
Layer 2: End-to-end pipeline test
"""

import math

import pytest
import torch

from tilegym.ops.nsa_reference import (
    compression_attention_ref,
    mean_pool_kv,
    nsa_forward_ref,
    selection_attention_ref,
    sliding_window_attention_ref,
    topk_block_selection_ref,
)


# ============================================================================
# Helper Functions
# ============================================================================


def make_causal_block_indices(B, G, S, block_count, block_size, device="cuda"):
    """Generate valid causal block indices for testing."""
    Tc = (S + block_size - 1) // block_size
    block_indices = torch.zeros(B, G, S, block_count, dtype=torch.long, device=device)

    for b in range(B):
        for g in range(G):
            for t in range(S):
                nc = (t + 1) // block_size  # number of accessible blocks
                if nc == 0:
                    # No valid blocks; fill with 0 (will be masked by causal)
                    block_indices[b, g, t] = 0
                else:
                    # Select min(block_count, nc) blocks randomly from [0, nc)
                    actual_k = min(block_count, nc)
                    perm = torch.randperm(nc, device=device)[:actual_k]
                    block_indices[b, g, t, :actual_k] = perm
                    # Pad remaining with first valid index
                    if actual_k < block_count:
                        block_indices[b, g, t, actual_k:] = perm[0]

    return block_indices


# ============================================================================
# Layer 0: Reference Cross-Validation
# ============================================================================


class TestReferenceCorrectness:
    """Cross-validate reference functions for internal consistency."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 1, 1, 128, 64),
            (1, 4, 1, 128, 64),
            (1, 4, 2, 128, 64),
        ],
    )
    @pytest.mark.parametrize("block_size", [32, 64])
    def test_mean_pool(self, B, HQ, G, S, D, block_size):
        """Test mean pooling produces correct shapes and values."""
        torch.manual_seed(42)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)

        k_cmp, v_cmp = mean_pool_kv(k, v, block_size)

        Tc = (S + block_size - 1) // block_size
        assert k_cmp.shape == (B, G, Tc, D)
        assert v_cmp.shape == (B, G, Tc, D)

        # Verify first block is mean of first block_size elements
        expected_k0 = k[:, :, :block_size].mean(dim=2)
        torch.testing.assert_close(k_cmp[:, :, 0], expected_k0, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 1, 1, 128, 64),
            (1, 4, 2, 128, 64),
        ],
    )
    @pytest.mark.parametrize("block_size", [32, 64])
    def test_compression_attention_causal(self, B, HQ, G, S, D, block_size):
        """Test compression attention respects block-level causal mask."""
        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)

        k_cmp, v_cmp = mean_pool_kv(k, v, block_size)
        o_cmp, lse_cmp = compression_attention_ref(q, k_cmp, v_cmp, block_size)

        assert o_cmp.shape == (B, HQ, S, D)
        assert lse_cmp.shape == (B, HQ, S)

        # First block_size-1 positions should have zero output
        # (NC=0, no valid compressed blocks)
        o_early = o_cmp[:, :, :block_size - 1]
        assert torch.allclose(o_early, torch.zeros_like(o_early), atol=1e-6)

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 1, 1, 128, 64),
            (1, 4, 2, 256, 64),
        ],
    )
    @pytest.mark.parametrize("block_size", [32, 64])
    @pytest.mark.parametrize("block_count", [4, 8])
    def test_selection_attention_vs_fla(self, B, HQ, G, S, D, block_size, block_count):
        """Cross-validate selection_attention_ref against FLA naive_nsa."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "fla_nsa_naive",
            "/root/workspace/flash-linear-attention/fla/ops/nsa/naive.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fla_naive_nsa = mod.naive_nsa

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)
        scale = D ** -0.5

        # Generate valid causal block indices
        block_indices = make_causal_block_indices(B, G, S, block_count, block_size, device="cuda")

        # Our reference: head-first layout
        o_ours = selection_attention_ref(q, k, v, block_indices, block_size, scale)

        # FLA: seq-first layout [B, T, H, D]
        q_fla = q.transpose(1, 2)  # [B, S, HQ, D]
        k_fla = k.transpose(1, 2)  # [B, S, G, D]
        v_fla = v.transpose(1, 2)  # [B, S, G, D]
        bi_fla = block_indices.transpose(1, 2)  # [B, S, G, block_count]

        o_fla = fla_naive_nsa(q_fla, k_fla, v_fla, bi_fla,
                              block_size=block_size, scale=scale, head_first=False)
        o_fla = o_fla.transpose(1, 2)  # [B, HQ, S, D]

        torch.testing.assert_close(
            o_ours.float(), o_fla.float(),
            atol=1e-3, rtol=1e-3,
            msg="selection_attention_ref vs FLA naive_nsa mismatch",
        )

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 1, 1, 128, 64),
            (1, 4, 2, 128, 64),
        ],
    )
    @pytest.mark.parametrize("window_size", [32, 64])
    def test_sliding_window_vs_pytorch_sdpa(self, B, HQ, G, S, D, window_size):
        """Cross-validate sliding window against manual implementation."""
        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)
        scale = D ** -0.5

        o_swa = sliding_window_attention_ref(q, k, v, window_size, scale)

        # Manual verification: for a specific position, check attention window
        QGS = HQ // G
        k_exp = k.unsqueeze(2).expand(-1, -1, QGS, -1, -1).reshape(B, HQ, S, D)
        v_exp = v.unsqueeze(2).expand(-1, -1, QGS, -1, -1).reshape(B, HQ, S, D)

        t = S - 1  # Last position
        lo = max(0, t - window_size + 1)
        q_t = q[0, 0, t] * scale
        k_win = k_exp[0, 0, lo:t + 1]
        v_win = v_exp[0, 0, lo:t + 1]
        attn = torch.softmax(q_t @ k_win.T, dim=-1)
        expected = attn @ v_win

        torch.testing.assert_close(o_swa[0, 0, t], expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 4, 2, 128, 64),
        ],
    )
    def test_selection_sliding_vs_tilelang(self, B, HQ, G, S, D):
        """Cross-validate selection + sliding window with gates against TileLang reference."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "tilelang_nsa_ref",
            "/root/workspace/tilelang/examples/deepseek_nsa/reference.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tilelang_naive_nsa = mod.naive_nsa

        torch.manual_seed(42)
        block_size = 32
        block_count = 4
        window_size = 64
        scale = D ** -0.5

        q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)
        g_slc = torch.randn(B, HQ, S, device="cuda", dtype=torch.float32)
        g_swa = torch.randn(B, HQ, S, device="cuda", dtype=torch.float32)

        block_indices = make_causal_block_indices(B, G, S, block_count, block_size, device="cuda")

        # Our reference
        o_slc = selection_attention_ref(q, k, v, block_indices, block_size, scale)
        o_swa = sliding_window_attention_ref(q, k, v, window_size, scale)
        # TileLang applies gates directly (no sigmoid), so pass post-sigmoid values
        g_slc_post = torch.sigmoid(g_slc)
        g_swa_post = torch.sigmoid(g_swa)
        o_ours = g_slc_post.unsqueeze(-1) * o_slc.float() + g_swa_post.unsqueeze(-1) * o_swa.float()

        # TileLang: seq-first layout, expects post-sigmoid gate values
        q_tl = q.transpose(1, 2)
        k_tl = k.transpose(1, 2)
        v_tl = v.transpose(1, 2)
        g_slc_tl = g_slc_post.transpose(1, 2)  # [B, S, HQ]
        g_swa_tl = g_swa_post.transpose(1, 2)
        bi_tl = block_indices.transpose(1, 2)

        o_tl = tilelang_naive_nsa(
            q_tl, k_tl, v_tl,
            g_slc=g_slc_tl, g_swa=g_swa_tl,
            block_indices=bi_tl,
            block_size=block_size,
            window_size=window_size,
            scale=scale,
            head_first=False,
        )
        o_tl = o_tl.transpose(1, 2)  # [B, HQ, S, D]

        torch.testing.assert_close(
            o_ours.float(), o_tl.float(),
            atol=1e-3, rtol=1e-3,
            msg="selection+sliding vs TileLang naive_nsa mismatch",
        )

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 4, 2, 128, 64),
        ],
    )
    def test_nsa_forward_ref_smoke(self, B, HQ, G, S, D):
        """Smoke test: full NSA forward ref produces correct shape."""
        torch.manual_seed(42)
        block_size = 32
        block_count = 4
        window_size = 64

        q = torch.randn(B, HQ, S, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, G, S, D, device="cuda", dtype=torch.float32)
        g_cmp = torch.randn(B, HQ, S, device="cuda", dtype=torch.float32)
        g_slc = torch.randn(B, HQ, S, device="cuda", dtype=torch.float32)
        g_swa = torch.randn(B, HQ, S, device="cuda", dtype=torch.float32)

        o = nsa_forward_ref(q, k, v, g_cmp, g_slc, g_swa,
                            block_size, block_count, window_size)

        assert o.shape == (B, HQ, S, D)
        assert not torch.isnan(o).any(), "NaN in NSA forward output"


# ============================================================================
# Layer 1: CuTile Kernel Unit Tests (populated in Phase 2-5)
# ============================================================================


class TestCompressionAttentionKernel:
    """Unit tests for compression attention cutile kernel."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 1, 1, 128, 64),
            (1, 16, 1, 256, 128),
            (2, 32, 2, 512, 128),
        ],
    )
    @pytest.mark.parametrize("block_size", [32, 64])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_compression_attn(self, B, HQ, G, S, D, block_size, dtype):
        """Test cutile compression attention kernel vs reference."""
        from tilegym.ops.cutile.nsa import compression_attention

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        scale = D ** -0.5

        k_cmp, v_cmp = mean_pool_kv(k.float(), v.float(), block_size)
        k_cmp = k_cmp.to(dtype)
        v_cmp = v_cmp.to(dtype)

        # Reference (float32)
        o_ref, lse_ref = compression_attention_ref(
            q.float(), k_cmp.float(), v_cmp.float(), block_size, scale)

        # CuTile kernel
        o_ct, lse_ct = compression_attention(q, k_cmp, v_cmp, block_size, scale)

        torch.testing.assert_close(
            o_ct.float(), o_ref.float(), atol=5e-2, rtol=1e-2,
            msg=f"compression_attn output mismatch (dtype={dtype})")


class TestSlidingWindowKernel:
    """Unit tests for sliding window cutile kernel."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 1, 1, 128, 64),
            (1, 16, 1, 256, 128),
            (2, 32, 2, 512, 128),
        ],
    )
    @pytest.mark.parametrize("window_size", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_sliding_window(self, B, HQ, G, S, D, window_size, dtype):
        """Test cutile sliding window kernel vs reference."""
        from tilegym.ops.cutile.nsa import sliding_window_attention

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        scale = D ** -0.5

        # Reference (float32)
        o_ref = sliding_window_attention_ref(q.float(), k.float(), v.float(), window_size, scale)

        # CuTile kernel
        o_ct = sliding_window_attention(q, k, v, window_size, scale)

        torch.testing.assert_close(
            o_ct.float(), o_ref.float(), atol=5e-2, rtol=1e-2,
            msg=f"sliding_window output mismatch (dtype={dtype})")


class TestSelectionAttentionKernel:
    """Unit tests for selection attention (PyTorch fallback)."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 1, 1, 128, 64),
            (1, 4, 2, 128, 64),
        ],
    )
    @pytest.mark.parametrize("block_size", [32, 64])
    @pytest.mark.parametrize("block_count", [4, 8])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_selection_attn(self, B, HQ, G, S, D, block_size, block_count, dtype):
        """Test selection attention (currently PyTorch fallback) vs reference."""
        from tilegym.ops.cutile.nsa import selection_attention

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        scale = D ** -0.5

        block_indices = make_causal_block_indices(B, G, S, block_count, block_size, device="cuda")

        # Both use the same reference implementation (fallback)
        o_ref = selection_attention_ref(q.float(), k.float(), v.float(),
                                        block_indices, block_size, scale)
        o_ct = selection_attention(q, k, v, block_indices, block_size, scale)

        torch.testing.assert_close(
            o_ct.float(), o_ref.float(), atol=5e-2, rtol=1e-2,
            msg=f"selection_attn output mismatch (dtype={dtype})")


# ============================================================================
# Layer 2: End-to-End Pipeline Tests (populated in Phase 6)
# ============================================================================


class TestNSAEndToEnd:
    """End-to-end tests for the full NSA pipeline."""

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [
            (1, 4, 2, 128, 64),
            (1, 16, 1, 256, 128),
        ],
    )
    @pytest.mark.parametrize("block_size", [32, 64])
    @pytest.mark.parametrize("block_count", [4])
    @pytest.mark.parametrize("window_size", [64])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_nsa_forward(self, B, HQ, G, S, D, block_size, block_count, window_size, dtype):
        """Test full tile_nsa vs nsa_forward_ref."""
        from tilegym.ops.cutile.nsa import tile_nsa

        torch.manual_seed(42)
        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        g_cmp = torch.randn(B, HQ, S, device="cuda", dtype=dtype)
        g_slc = torch.randn(B, HQ, S, device="cuda", dtype=dtype)
        g_swa = torch.randn(B, HQ, S, device="cuda", dtype=dtype)
        scale = D ** -0.5

        # Reference (float32)
        o_ref = nsa_forward_ref(
            q.float(), k.float(), v.float(),
            g_cmp.float(), g_slc.float(), g_swa.float(),
            block_size, block_count, window_size, scale,
        )

        # CuTile pipeline
        o_ct = tile_nsa(
            q, k, v, g_cmp, g_slc, g_swa,
            block_size=block_size, block_count=block_count,
            window_size=window_size, scale=scale,
        )

        torch.testing.assert_close(
            o_ct.float(), o_ref.float(), atol=5e-2, rtol=1e-2,
            msg=f"NSA E2E mismatch (dtype={dtype}, B={B}, HQ={HQ}, G={G}, S={S}, D={D})"
        )
