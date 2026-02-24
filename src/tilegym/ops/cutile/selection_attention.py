# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CuTile selection attention: attend to selected K/V blocks per query position.

Uses ct.gather for data-dependent K/V loading from block_indices, with online
softmax for numerical stability across all selected blocks.

Algorithm:
  For each query tile [bid_x * TILE_M : (bid_x+1) * TILE_M]:
    For each selected block b in 0..block_count-1:
      For each sub-tile of block_size tokens:
        1. ct.gather K rows at block_indices[b] * block_size + sub_offset
        2. Compute QK scores via element-wise multiply + reduce
        3. Online softmax update (max tracking, rescaling)
        4. ct.gather V rows, accumulate weighted output

Note: Cannot use ct.mma for QK because each query position has different
block indices, making the K tile 3D [TILE_M, TILE_N, TILE_D] rather than
2D [TILE_N, TILE_D]. Element-wise multiply + sum is used instead.

Grid: (ceil(S / TILE_M), B * HQ, 1)
"""

import math
import os

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl
from tilegym.logger import get_logger
from tilegym.ops.nsa_reference import selection_attention_ref

from .utils import next_power_of_2

logger = get_logger(__name__)

INV_LOG_2 = 1.0 / math.log(2)

ConstInt = ct.Constant[int]


def _should_disable_autotune():
    return os.environ.get("DISABLE_AUTOTUNE", "0") == "1"


# ============================================================================
# CuTile Kernel: Selection Attention via ct.gather + online softmax
# ============================================================================


@ct.kernel
def selection_attention_kernel(
    Q,              # [B, HQ, S, D]
    K,              # [B, G, S, D]
    V,              # [B, G, S, D]
    BlockIndices,   # [B, G, S, block_count] int64
    Out,            # [B, HQ, S, D]
    qk_scale: float,
    TILE_D: ConstInt,
    H: ConstInt,
    BLOCK_SIZE: ConstInt,
    BLOCK_COUNT: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    SEQ_LEN: ConstInt,
):
    """
    Selection attention kernel with data-dependent K/V gathering.

    Each thread block processes one query tile of TILE_M positions
    for one (batch, head) combination.
    """
    bid_x = ct.bid(0)  # query tile index
    bid_y = ct.bid(1)  # batch * HQ + head

    batch_idx = bid_y // H
    head_idx = bid_y % H
    kv_head = head_idx // QUERY_GROUP_SIZE

    qk_scale_log2 = qk_scale * INV_LOG_2

    # Query position offsets for this tile: [TILE_M]
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)

    # Online softmax accumulators
    m_i = ct.full((TILE_M, 1), -1.0e30, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Load query tile: [TILE_M, D]
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_M, TILE_D),
                padding_mode=ct.PaddingMode.ZERO).reshape((TILE_M, TILE_D))

    kv_S = K.shape[2]

    # Indices for D dimension: [TILE_D]
    d_indices = ct.arange(TILE_D, dtype=ct.int32)

    # Iterate over selected blocks
    for blk in range(BLOCK_COUNT):
        # Gather block indices: BlockIndices[batch, kv_head, query_pos, blk]
        # offs_m is [TILE_M], rest are scalars → broadcasts to [TILE_M]
        block_idx = ct.gather(
            BlockIndices,
            (batch_idx, kv_head, offs_m, blk),
        )  # [TILE_M] int64

        # Base KV position for each query's selected block: [TILE_M]
        kv_base = block_idx * BLOCK_SIZE

        # Iterate over sub-tiles within this block
        n_subtiles = ct.cdiv(BLOCK_SIZE, TILE_N)
        for sub in range(n_subtiles):
            sub_offset = sub * TILE_N

            # KV position offsets within sub-tile: [TILE_N]
            n_offsets = ct.arange(TILE_N, dtype=ct.int32)

            # Full KV positions: [TILE_M, TILE_N] via broadcast
            kv_pos = kv_base[:, None] + sub_offset + n_offsets[None, :]

            # Clamp for safe gather (OOB returns padding_value=0)
            kv_pos_safe = ct.where(kv_pos < kv_S, kv_pos, 0).astype(ct.int32)

            # Gather K: K[batch, kv_head, kv_pos, d] → [TILE_M, TILE_N, TILE_D]
            k_gathered = ct.gather(
                K,
                (batch_idx, kv_head, kv_pos_safe[:, :, None], d_indices[None, None, :]),
            )

            # QK scores: sum_d(q[m,d] * k[m,n,d]) → [TILE_M, TILE_N]
            qk = ct.sum(q[:, None, :] * k_gathered, axis=-1)

            # Causal mask: kv_pos > query_pos → invalid
            causal_invalid = kv_pos > offs_m[:, None]
            # OOB mask: kv_pos >= kv_S or sub_offset + n >= BLOCK_SIZE
            oob = (kv_pos >= kv_S) | (sub_offset + n_offsets[None, :] >= BLOCK_SIZE)
            invalid = causal_invalid | oob
            qk = qk + ct.where(invalid, -1.0e6, 0.0)

            # Online softmax update
            m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
            qk = qk * qk_scale_log2 - m_ij

            p = ct.exp2(qk, flush_to_zero=True)
            l_ij = ct.sum(p, axis=-1, keepdims=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha

            # Gather V: V[batch, kv_head, kv_pos, d] → [TILE_M, TILE_N, TILE_D]
            v_gathered = ct.gather(
                V,
                (batch_idx, kv_head, kv_pos_safe[:, :, None], d_indices[None, None, :]),
            )

            # Weighted accumulation: acc[m,d] += sum_n(p[m,n] * v[m,n,d])
            acc = acc + ct.sum(p[:, :, None] * v_gathered, axis=1)

            m_i = m_ij

    # Normalize
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


# ============================================================================
# Python Wrapper
# ============================================================================


def selection_attention(q, k, v, block_indices, block_size, block_count=None, scale=None):
    """
    Selection attention: attend to selected K/V blocks per query position.

    Uses CuTile kernel with ct.gather for data-dependent K/V loading.
    Falls back to PyTorch reference for unsupported configurations.

    Args:
        q: [B, HQ, S, D]
        k: [B, G, S, D]
        v: [B, G, S, D]
        block_indices: [B, G, S, block_count]
        block_size: int
        block_count: int (unused, inferred from block_indices)
        scale: float

    Returns:
        o_slc: [B, HQ, S, D]
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    block_indices = block_indices.contiguous()

    B, HQ, S, D = q.shape
    _, G, _, _ = k.shape
    actual_block_count = block_indices.shape[-1]
    QGS = HQ // G

    if scale is None:
        scale = D ** -0.5

    # Tile sizes — must be powers of 2
    TILE_D = next_power_of_2(D)
    TILE_M = 64  # query positions per tile
    TILE_N = min(32, next_power_of_2(block_size))  # KV sub-tile size

    # Fallback to reference for non-power-of-2 D or other unsupported configs
    if D != TILE_D:
        logger.debug("Falling back to reference: D=%d is not power of 2", D)
        return selection_attention_ref(q, k, v, block_indices, block_size, scale)

    output = torch.empty(B, HQ, S, D, device=q.device, dtype=q.dtype)

    grid = (math.ceil(S / TILE_M), B * HQ, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        selection_attention_kernel,
        (
            q, k, v, block_indices, output,
            scale,
            TILE_D, HQ, block_size, actual_block_count,
            TILE_M, TILE_N, QGS, S,
        ),
    )

    return output


# Register with dispatcher
register_impl("selection_attention", "cutile")(selection_attention)
