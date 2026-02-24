# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CuTile kernels and Python wrappers for NSA (Native Sparse Attention) forward pass.

Architecture:
  Input: q[B,HQ,S,D], k[B,G,S,D], v[B,G,S,D], gates[B,HQ,S,3]
    ├─ Stage 1: Mean Pool (PyTorch) → k_cmp, v_cmp
    ├─ Stage 2: Compression Attention (cutile kernel) → o_cmp, lse_cmp
    ├─ Stage 3: Top-K Selection (PyTorch) → block_indices
    ├─ Stage 4: Selection Attention (PyTorch fallback) → o_slc
    ├─ Stage 5: Sliding Window Attention (from sliding_window_attention.py) → o_swa
    └─ Stage 6: Gate Fusion (PyTorch) → o
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
from tilegym.ops.nsa_reference import (
    mean_pool_kv,
    selection_attention_ref,
)

from .sliding_window_attention import sliding_window_attention
from .utils import next_power_of_2

logger = get_logger(__name__)

INV_LOG_2 = 1.0 / math.log(2)
LN2 = math.log(2)

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def _should_disable_autotune():
    return os.environ.get("DISABLE_AUTOTUNE", "0") == "1"


# ============================================================================
# Kernel 1: Compression Attention
# Adapted from fmha_fwd_kernel_with_lse
# Key difference: KV seq_len is Tc (compressed), block-level causal mask
# ============================================================================


@ct.kernel(occupancy=2)
def nsa_compression_attn_kernel(
    Q,      # [B, HQ, S, D]
    K_cmp,  # [B, G, Tc, D]
    V_cmp,  # [B, G, Tc, D]
    Out,    # [B, HQ, S, D]
    LSE,    # 1D flattened [B * HQ * padded_S]
    qk_scale: float,
    TILE_D: ConstInt,
    H: ConstInt,
    B: ConstInt,
    SEQ_LEN: ConstInt,
    BLOCK_SIZE: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
):
    """
    Compression attention kernel with block-level causal masking.

    Each query at position t can attend to compressed block c
    only if c < (t+1) // BLOCK_SIZE.
    """
    bid_x = ct.bid(0)  # query tile index
    bid_y = ct.bid(1)  # batch * HQ + head
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    qk_scale = qk_scale * INV_LOG_2

    # Query tile offsets
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

    # Compressed KV sequence length
    kv_seqlen = K_cmp.shape[2]  # Tc

    # Block-level causal bound: max accessible compressed block index
    # For query tile starting at bid_x * TILE_M, the last query is at
    # bid_x * TILE_M + TILE_M - 1. Its NC = (pos + 1) // BLOCK_SIZE.
    # We iterate over all blocks up to the max NC in this tile.
    max_query_pos = bid_x * TILE_M + TILE_M - 1
    max_nc = (max_query_pos + 1) // BLOCK_SIZE
    Tc = min(max_nc, kv_seqlen)

    # Determine where masking starts (all blocks before this are fully accessible)
    min_query_pos = bid_x * TILE_M
    min_nc = (min_query_pos + 1) // BLOCK_SIZE
    mask_start = min(min_nc, kv_seqlen // TILE_N)

    for j in range(0, ct.cdiv(Tc, TILE_N)):
        # Load compressed K transposed
        k = ct.load(
            K_cmp,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        ).reshape((TILE_D, TILE_N))

        # QK product
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        # Block-level causal mask
        offs_n = j * TILE_N + offs_n_tile  # actual compressed block indices
        # NC for each query = (query_pos + 1) // BLOCK_SIZE
        nc = (offs_m + 1) // BLOCK_SIZE  # [TILE_M, 1]
        # Use -1e6 instead of -inf to avoid NaN in online softmax
        oob = offs_n >= kv_seqlen
        not_causal = offs_n >= nc  # block c must satisfy c < nc
        invalid = oob | not_causal
        qk = qk + ct.where(invalid, -1.0e6, 0.0)

        # Online softmax update
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Load compressed V
        v = ct.load(
            V_cmp,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Normalize
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    # Store LSE
    lse_tile = m_i + ct.log2(l_i)
    lse_tile = lse_tile.reshape((TILE_M,))
    lse_offsets = ct.arange(TILE_M, dtype=ct.int32)
    lse_indices = batch_idx * (H * SEQ_LEN) + head_idx * SEQ_LEN + bid_x * TILE_M + lse_offsets
    ct.scatter(LSE, lse_indices, lse_tile)


# ============================================================================
# Sliding Window Attention: extracted to sliding_window_attention.py
# Imported above as: from .sliding_window_attention import sliding_window_attention
# ============================================================================


# ============================================================================
# Autotune Configurations
# ============================================================================

_NSA_FWD_TILE_CONFIGS_BY_D = {
    64: ([64, 128, 256], [32, 64, 128]),
    128: ([64, 128, 256], [32, 64, 128]),
    256: ([64, 128], [32, 64]),
}


def _nsa_autotune_configs(head_dim=None):
    key = next_power_of_2(head_dim) if head_dim else 128
    tile_ms, tile_ns = _NSA_FWD_TILE_CONFIGS_BY_D.get(key, ([64, 128, 256], [32, 64, 128]))
    for tm in tile_ms:
        for tn in tile_ns:
            yield SimpleNamespace(TILE_M=tm, TILE_N=tn)


# ============================================================================
# Python Wrappers
# ============================================================================


def compression_attention(q, k_cmp, v_cmp, block_size, scale=None):
    """
    Compression attention using cutile kernel.

    Args:
        q: [B, HQ, S, D]
        k_cmp: [B, G, Tc, D]
        v_cmp: [B, G, Tc, D]
        block_size: int
        scale: float

    Returns:
        o_cmp: [B, HQ, S, D]
        lse_cmp: [B, HQ, S]
    """
    q = q.contiguous()
    k_cmp = k_cmp.contiguous()
    v_cmp = v_cmp.contiguous()

    B, HQ, S, D = q.shape
    _, G, Tc, _ = k_cmp.shape

    assert HQ % G == 0
    query_group_size = HQ // G

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    o = torch.empty_like(q)

    configs = list(_nsa_autotune_configs(D))
    cfg = configs[0]
    TILE_M = cfg.TILE_M
    padded_S = math.ceil(S / TILE_M) * TILE_M
    lse_flat = torch.zeros(B * HQ * padded_S, dtype=torch.float32, device=q.device)

    stream = torch.cuda.current_stream()

    if _should_disable_autotune():
        grid = (math.ceil(S / cfg.TILE_M), B * HQ, 1)
        ct.launch(
            stream, grid,
            nsa_compression_attn_kernel,
            (q, k_cmp, v_cmp, o, lse_flat, scale, D, HQ,
             B, padded_S, block_size, cfg.TILE_M, cfg.TILE_N, query_group_size),
        )
    else:
        ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (math.ceil(S / cfg.TILE_M), B * HQ, 1),
            kernel=nsa_compression_attn_kernel,
            args_fn=lambda cfg: (
                q, k_cmp, v_cmp, o, lse_flat, scale, D, HQ,
                B, padded_S, block_size, cfg.TILE_M, cfg.TILE_N, query_group_size,
            ),
            search_space=lambda: _nsa_autotune_configs(D),
            max_iter=20,
        )

    # Zero out positions with no valid compressed blocks (NC=0, i.e., pos < block_size)
    # These positions have uniform softmax over -1e6 values which produces small
    # non-zero artifacts. Semantically they should be zero.
    if block_size > 1:
        o[:, :, :block_size - 1] = 0

    lse = lse_flat.view(B, HQ, padded_S)[:, :, :S].contiguous()
    return o, lse


# sliding_window_attention is imported from .sliding_window_attention


# ============================================================================
# Stage 3: Top-K Selection (PyTorch — no cutile sort primitives)
# ============================================================================


def compute_importance_and_select(q, k_cmp, lse_cmp, block_size, block_count, scale=None):
    """
    Compute importance scores and select top-K blocks per query position.

    Args:
        q: [B, HQ, S, D]
        k_cmp: [B, G, Tc, D]
        lse_cmp: [B, HQ, S]
        block_size: int
        block_count: int
        scale: float

    Returns:
        block_indices: [B, G, S, block_count]
    """
    B, HQ, S, D = q.shape
    _, G, Tc, _ = k_cmp.shape
    QGS = HQ // G

    if scale is None:
        scale = D ** -0.5

    # Group queries by KV head
    q_grouped = q.float().reshape(B, G, QGS, S, D)
    k_f = k_cmp.float()

    # Scores: [B, G, QGS, S, Tc]
    scores = torch.einsum("bgqsd,bgtd->bgqst", q_grouped, k_f) * scale

    # importance = sum over query heads of exp(scores - lse)
    lse_grouped = lse_cmp.reshape(B, G, QGS, S).unsqueeze(-1)
    importance = torch.exp(scores - lse_grouped)
    importance = importance.sum(dim=2)  # [B, G, S, Tc]

    # Block-level causal mask
    query_pos = torch.arange(S, device=q.device).view(1, 1, S, 1)
    block_idx = torch.arange(Tc, device=q.device).view(1, 1, 1, Tc)
    causal_mask = block_idx < (query_pos + 1) // block_size
    importance = importance.masked_fill(~causal_mask, float("-inf"))

    # Top-K
    actual_k = min(block_count, Tc)
    _, indices = torch.topk(importance, k=actual_k, dim=-1)

    if actual_k < block_count:
        pad_indices = torch.zeros(
            B, G, S, block_count - actual_k,
            dtype=indices.dtype, device=indices.device,
        )
        indices = torch.cat([indices, pad_indices], dim=-1)

    return indices


# ============================================================================
# Stage 4: Selection Attention (PyTorch fallback)
# Uses reference implementation — data-dependent ct.load not yet verified
# ============================================================================


def selection_attention(q, k, v, block_indices, block_size, scale=None):
    """
    Selection attention using PyTorch fallback.

    This uses the reference implementation since data-dependent ct.load
    (indirect indexing) has not been verified in cutile yet.

    Args:
        q: [B, HQ, S, D]
        k: [B, G, S, D]
        v: [B, G, S, D]
        block_indices: [B, G, S, block_count]
        block_size: int
        scale: float

    Returns:
        o_slc: [B, HQ, S, D]
    """
    return selection_attention_ref(q, k, v, block_indices, block_size, scale)


# ============================================================================
# Stage 6: Gate Fusion (PyTorch)
# ============================================================================


def gate_fusion(o_cmp, o_slc, o_swa, g_cmp, g_slc, g_swa):
    """
    Gate-weighted fusion of three attention outputs.

    o = sigmoid(g_cmp) * o_cmp + sigmoid(g_slc) * o_slc + sigmoid(g_swa) * o_swa
    """
    g_cmp_s = torch.sigmoid(g_cmp.float()).unsqueeze(-1)
    g_slc_s = torch.sigmoid(g_slc.float()).unsqueeze(-1)
    g_swa_s = torch.sigmoid(g_swa.float()).unsqueeze(-1)
    return (g_cmp_s * o_cmp.float() + g_slc_s * o_slc.float() + g_swa_s * o_swa.float())


# ============================================================================
# Full Pipeline
# ============================================================================


def tile_nsa(q, k, v, g_cmp, g_slc, g_swa,
             block_size=64, block_count=16, window_size=512, scale=None, **kwargs):
    """
    NSA forward pass combining compression, selection, and sliding window attention.

    Args:
        q: [B, HQ, S, D]
        k: [B, G, S, D]
        v: [B, G, S, D]
        g_cmp: [B, HQ, S] — compression gate (pre-sigmoid)
        g_slc: [B, HQ, S] — selection gate (pre-sigmoid)
        g_swa: [B, HQ, S] — sliding window gate (pre-sigmoid)
        block_size: int
        block_count: int
        window_size: int
        scale: float

    Returns:
        o: [B, HQ, S, D]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # Stage 1: Mean Pool (PyTorch)
    k_cmp, v_cmp = mean_pool_kv(k, v, block_size)

    # Stage 2: Compression Attention (cutile)
    o_cmp, lse_cmp = compression_attention(q, k_cmp, v_cmp, block_size, scale)

    # Stage 3: Top-K Selection (PyTorch)
    block_indices = compute_importance_and_select(
        q, k_cmp, lse_cmp, block_size, block_count, scale)

    # Stage 4: Selection Attention (PyTorch fallback)
    o_slc = selection_attention(q, k, v, block_indices, block_size, scale)

    # Stage 5: Sliding Window Attention (cutile)
    o_swa = sliding_window_attention(q, k, v, window_size, scale)

    # Stage 6: Gate Fusion
    o = gate_fusion(o_cmp, o_slc, o_swa, g_cmp, g_slc, g_swa)
    return o.to(q.dtype)


# Register with dispatcher
register_impl("nsa", "cutile")(tile_nsa)
