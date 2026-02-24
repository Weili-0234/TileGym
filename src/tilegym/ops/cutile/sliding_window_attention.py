# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Standalone CuTile sliding window attention kernel.

Extracted from nsa.py for independent use. Implements causal sliding window
attention where each query at position t attends to positions [max(0, t-W+1), t].

Tensor layout: [B, HQ, S, D] (head-first)
Supports GQA: HQ query heads, G KV heads (HQ = G * QUERY_GROUP_SIZE)
"""

import math
import os
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl
from tilegym.logger import get_logger

from .utils import next_power_of_2

logger = get_logger(__name__)

INV_LOG_2 = 1.0 / math.log(2)

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def _should_disable_autotune():
    return os.environ.get("DISABLE_AUTOTUNE", "0") == "1"


# ============================================================================
# Autotune Configurations
# ============================================================================

_SWA_TILE_CONFIGS_BY_D = {
    64: ([64, 128, 256], [32, 64, 128]),
    128: ([64, 128, 256], [32, 64, 128]),
    256: ([64, 128], [32, 64]),
}


def _swa_autotune_configs(head_dim=None):
    key = next_power_of_2(head_dim) if head_dim else 128
    tile_ms, tile_ns = _SWA_TILE_CONFIGS_BY_D.get(key, ([64, 128, 256], [32, 64, 128]))
    for tm in tile_ms:
        for tn in tile_ns:
            yield SimpleNamespace(TILE_M=tm, TILE_N=tn)


# ============================================================================
# Kernel: Sliding Window Attention
# ============================================================================


@ct.kernel(occupancy=2)
def sliding_window_attention_kernel(
    Q,    # [B, HQ, S, D]
    K,    # [B, G, S, D]
    V,    # [B, G, S, D]
    Out,  # [B, HQ, S, D]
    qk_scale: float,
    TILE_D: ConstInt,
    H: ConstInt,
    SEQ_LEN: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    WINDOW_SIZE: ConstInt,
):
    """
    Sliding window causal attention kernel.

    Each query at position t attends to positions [max(0, t-W+1), t].
    """
    bid_x = ct.bid(0)  # query tile index
    bid_y = ct.bid(1)  # batch * HQ + head
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    qk_scale = qk_scale * INV_LOG_2

    # Query offsets
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # KV tile offsets
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Online softmax accumulators
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Load query tile
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    k_seqlen = K.shape[2]

    # Window bounds (from attention_sink.py pattern)
    # lo = max(0, query_tile_start - WINDOW_SIZE + 1)
    # hi = query_tile_end (causal)
    lo = ct.maximum(0, bid_x * TILE_M - WINDOW_SIZE + 1)
    hi = min((bid_x + 1) * TILE_M, k_seqlen)

    start_block = lo // TILE_N
    Tc = ct.cdiv(hi, TILE_N)

    for j in range(start_block, Tc):
        offs_n = j * TILE_N + offs_n_tile

        # Load K transposed
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        ).reshape((TILE_D, TILE_N))

        # QK product
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        # Combined causal + window + OOB mask
        # Use -1e6 instead of -inf to avoid NaN in online softmax when entire
        # blocks fall outside the window (same pattern as attention_sink.py)
        oob_mask = offs_n >= k_seqlen
        causal_mask = offs_n > offs_m
        too_old = offs_n < (offs_m - WINDOW_SIZE + 1)
        invalid = oob_mask | causal_mask | too_old
        qk = qk + ct.where(invalid, -1.0e6, 0.0)

        # Online softmax
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Load V
        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Normalize and store
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


# ============================================================================
# Python Wrapper
# ============================================================================


def sliding_window_attention(q, k, v, window_size, scale=None):
    """
    Sliding window attention using CuTile kernel.

    Args:
        q: [B, HQ, S, D]
        k: [B, G, S, D]
        v: [B, G, S, D]
        window_size: int
        scale: float

    Returns:
        o_swa: [B, HQ, S, D]
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, HQ, S, D = q.shape
    _, G, _, _ = k.shape

    assert HQ % G == 0
    query_group_size = HQ // G

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    o = torch.empty_like(q)
    stream = torch.cuda.current_stream()

    if _should_disable_autotune():
        configs = list(_swa_autotune_configs(D))
        cfg = configs[0]
        grid = (math.ceil(S / cfg.TILE_M), B * HQ, 1)
        ct.launch(
            stream, grid,
            sliding_window_attention_kernel,
            (q, k, v, o, scale, D, HQ, S,
             cfg.TILE_M, cfg.TILE_N, query_group_size, window_size),
        )
    else:
        ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (math.ceil(S / cfg.TILE_M), B * HQ, 1),
            kernel=sliding_window_attention_kernel,
            args_fn=lambda cfg: (
                q, k, v, o, scale, D, HQ, S,
                cfg.TILE_M, cfg.TILE_N, query_group_size, window_size,
            ),
            search_space=lambda: _swa_autotune_configs(D),
            max_iter=20,
        )

    return o


# Register with dispatcher
register_impl("sliding_window_attention", "cutile")(sliding_window_attention)
