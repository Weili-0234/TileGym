# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym
from tests import common


class Test_SlidingWindowAttention(common.PyTestCase):
    @staticmethod
    def reference(q, k, v, window_size, scale=None):
        """PyTorch reference for sliding window causal attention."""
        bsz, hq, seqlen, dim = q.shape
        _, g, _, _ = k.shape
        qgs = hq // g

        if scale is None:
            scale = dim**-0.5

        k_exp = k.unsqueeze(2).expand(-1, -1, qgs, -1, -1).reshape(bsz, hq, seqlen, dim)
        v_exp = v.unsqueeze(2).expand(-1, -1, qgs, -1, -1).reshape(bsz, hq, seqlen, dim)

        q_f = q.float()
        k_f = k_exp.float()
        v_f = v_exp.float()

        scores = torch.einsum("bhsd,bhtd->bhst", q_f, k_f) * scale

        query_pos = torch.arange(seqlen, device=q.device).view(1, 1, seqlen, 1)
        key_pos = torch.arange(seqlen, device=q.device).view(1, 1, 1, seqlen)
        mask = (key_pos <= query_pos) & (key_pos >= query_pos - window_size + 1)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.einsum("bhst,bhtd->bhsd", attn, v_f)
        return out.to(q.dtype)

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "B, HQ, G, S, D",
        [(1, 8, 4, 256, 128), (1, 16, 1, 512, 64), (2, 4, 2, 128, 64), (1, 4, 2, 96, 64), (1, 64, 1, 256, 64)],
    )
    @pytest.mark.parametrize("window_size", [64, 128, 512])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, B, HQ, G, S, D, window_size, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        self.setUp()

        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        scale = D**-0.5

        self.assertCorrectness(
            tilegym.ops.sliding_window_attention,
            self.reference,
            {
                "q": q,
                "k": k,
                "v": v,
                "window_size": window_size,
                "scale": scale,
            },
            rtol=1e-2,
            atol=5e-2,
            check_stride=False,
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_window_larger_than_seq(self, dtype, backend):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        self.setUp()

        B, HQ, G, S, D = 1, 4, 2, 128, 64
        window_size = 256
        scale = D**-0.5

        q = torch.randn(B, HQ, S, D, device="cuda", dtype=dtype)
        k = torch.randn(B, G, S, D, device="cuda", dtype=dtype)
        v = torch.randn(B, G, S, D, device="cuda", dtype=dtype)

        self.assertCorrectness(
            tilegym.ops.sliding_window_attention,
            self.reference,
            {
                "q": q,
                "k": k,
                "v": v,
                "window_size": window_size,
                "scale": scale,
            },
            rtol=1e-2,
            atol=5e-2,
            check_stride=False,
        )
