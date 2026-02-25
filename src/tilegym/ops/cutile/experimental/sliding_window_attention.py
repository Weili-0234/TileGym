# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
import os
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl
from tilegym.experimental import experimental_kernel

from ..utils import next_power_of_2

INV_LOG_2 = 1.0 / math.log(2)

ConstInt = ct.Constant[int]


def _should_disable_autotune():
    return os.environ.get("DISABLE_AUTOTUNE", "0") == "1"


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


@experimental_kernel
@ct.kernel(occupancy=2)
def sliding_window_attention_kernel(
    Q,  # [B, HQ, S, D]
    K,  # [B, G, S, D]
    V,  # [B, G, S, D]
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
    """Sliding window causal attention kernel."""
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    qk_scale = qk_scale * INV_LOG_2

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]

    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]

    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    k_seqlen = K.shape[2]
    lo = ct.maximum(0, bid_x * TILE_M - WINDOW_SIZE + 1)
    hi = min((bid_x + 1) * TILE_M, k_seqlen)

    start_block = lo // TILE_N
    tc = ct.cdiv(hi, TILE_N)

    for j in range(start_block, tc):
        offs_n = j * TILE_N + offs_n_tile

        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        ).reshape((TILE_D, TILE_N))

        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        # Use a large negative finite value instead of -inf to avoid NaN in
        # online-softmax for fully-masked tiles.
        oob_mask = offs_n >= k_seqlen
        causal_mask = offs_n > offs_m
        too_old = offs_n < (offs_m - WINDOW_SIZE + 1)
        invalid = oob_mask | causal_mask | too_old
        qk = qk + ct.where(invalid, -1.0e6, 0.0)

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


def sliding_window_attention(q, k, v, window_size, scale=None):
    """Sliding window attention using CuTile kernel."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    bsz, hq, seqlen, head_dim = q.shape
    _, g, _, _ = k.shape

    assert hq % g == 0
    query_group_size = hq // g

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)
    stream = torch.cuda.current_stream()

    if _should_disable_autotune():
        configs = list(_swa_autotune_configs(head_dim))
        cfg = configs[0]
        grid = (math.ceil(seqlen / cfg.TILE_M), bsz * hq, 1)
        ct.launch(
            stream,
            grid,
            sliding_window_attention_kernel,
            (
                q,
                k,
                v,
                out,
                scale,
                head_dim,
                hq,
                seqlen,
                cfg.TILE_M,
                cfg.TILE_N,
                query_group_size,
                window_size,
            ),
        )
    else:
        ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (math.ceil(seqlen / cfg.TILE_M), bsz * hq, 1),
            kernel=sliding_window_attention_kernel,
            args_fn=lambda cfg: (
                q,
                k,
                v,
                out,
                scale,
                head_dim,
                hq,
                seqlen,
                cfg.TILE_M,
                cfg.TILE_N,
                query_group_size,
                window_size,
            ),
            search_space=lambda: _swa_autotune_configs(head_dim),
            max_iter=20,
        )

    return out


register_impl("sliding_window_attention", "cutile")(sliding_window_attention)
