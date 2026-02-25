# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

BATCH, HQ, G, HEAD_DIM = 4, 32, 4, 128


def reference_swa(q, k, v, window_size, scale=None):
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


register_impl("sliding_window_attention", "torch")(reference_swa)

# --- Flex Attention setup ---
try:
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    _compiled_flex_attention = torch.compile(_flex_attention)
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False

ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("flex", "FlexAttention", ("blue", "--")) if FLEX_AVAILABLE else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(datatype, window_size):
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(10, 15)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="TFLOPS",
        plot_name=f"swa-batch{BATCH}-hq{HQ}-g{G}-d{HEAD_DIM}-w{window_size}-{dtype_name}-TFLOPS",
        args={
            "datatype": datatype,
            "window_size": window_size,
        },
    )


_dtypes = [torch.float16, torch.bfloat16]
_window_sizes = [128, 512]


# Cache compiled block masks keyed by (window_size, N_CTX)
_block_mask_cache = {}


def _get_block_mask(window_size, N_CTX):
    key = (window_size, N_CTX)
    if key not in _block_mask_cache:

        def swa_mask(_b, _h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (kv_idx >= q_idx - window_size + 1)

        _block_mask_cache[key] = create_block_mask(swa_mask, BATCH, None, N_CTX, N_CTX, device=DEVICE)
    return _block_mask_cache[key]


@triton.testing.perf_report(
    [
        create_benchmark_config(datatype, window_size)
        for datatype in _dtypes
        for window_size in _window_sizes
    ]
)
def bench_sliding_window_attention(
    N_CTX,
    backend,
    datatype,
    window_size,
    device=DEVICE,
):
    scale = HEAD_DIM**-0.5

    q = torch.randn((BATCH, HQ, N_CTX, HEAD_DIM), dtype=datatype, device=device)
    k = torch.randn((BATCH, G, N_CTX, HEAD_DIM), dtype=datatype, device=device)
    v = torch.randn((BATCH, G, N_CTX, HEAD_DIM), dtype=datatype, device=device)

    if backend == "flex":
        block_mask = _get_block_mask(window_size, N_CTX)
        fn = lambda: _compiled_flex_attention(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=True)
    else:
        fn = lambda: tilegym.ops.sliding_window_attention(
            q, k, v, window_size, scale=scale, backend=backend
        )

    # Correctness check at small sequence lengths only (reference is O(S^2))
    if N_CTX <= 1024:
        ref_out = reference_swa(q, k, v, window_size, scale=scale)
        torch.testing.assert_close(fn(), ref_out, atol=1e-2, rtol=1e-2)

    try:
        ms = triton.testing.do_bench_cudagraph(fn)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan")

    # Exact sliding window causal FLOPS:
    # Each query at position s attends to min(W, s+1) keys
    # Total pairs = W*S - W*(W-1)/2  (when W <= S)
    W = min(window_size, N_CTX)
    total_pairs = W * N_CTX - W * (W - 1) // 2
    # 2*D FLOPs for QK matmul + 2*D FLOPs for PV matmul per pair
    total_flops = 4.0 * BATCH * HQ * total_pairs * HEAD_DIM
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench_sliding_window_attention.run(print_data=True)
