# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Pure PyTorch reference implementation of NSA (Native Sparse Attention) forward pass.

All tensors use TileGym head-first layout: [B, H, S, D]
- q: [B, HQ, S, D] (query heads)
- k, v: [B, G, S, D] (KV heads, G groups)
- HQ = G * QUERY_GROUP_SIZE

Reference for cross-validation against FLA naive and cutile kernels.
"""

import math

import torch


def mean_pool_kv(k, v, block_size):
    """
    Mean pool K and V along the sequence dimension by block_size.

    Args:
        k: [B, G, S, D]
        v: [B, G, S, D]
        block_size: int

    Returns:
        k_cmp: [B, G, Tc, D] where Tc = ceil(S / block_size)
        v_cmp: [B, G, Tc, D]
    """
    B, G, S, D = k.shape
    if S % block_size == 0:
        Tc = S // block_size
        k_cmp = k.reshape(B, G, Tc, block_size, D).mean(dim=3)
        v_cmp = v.reshape(B, G, Tc, block_size, D).mean(dim=3)
    else:
        Tc = (S + block_size - 1) // block_size
        k_parts = []
        v_parts = []
        for i in range(Tc):
            start = i * block_size
            end = min(start + block_size, S)
            k_parts.append(k[:, :, start:end].mean(dim=2))
            v_parts.append(v[:, :, start:end].mean(dim=2))
        k_cmp = torch.stack(k_parts, dim=2)
        v_cmp = torch.stack(v_parts, dim=2)
    return k_cmp, v_cmp


def compression_attention_ref(q, k_cmp, v_cmp, block_size, scale=None):
    """
    Compression attention: causal attention between Q and compressed K/V.

    Block-level causal: query at position t can attend to compressed block c
    only if c < (t+1) // block_size (from FLA: NC = (i_t + 1) // BS).

    Args:
        q: [B, HQ, S, D]
        k_cmp: [B, G, Tc, D]
        v_cmp: [B, G, Tc, D]
        block_size: int
        scale: float, default 1/sqrt(D)

    Returns:
        o_cmp: [B, HQ, S, D]
        lse_cmp: [B, HQ, S]
    """
    B, HQ, S, D = q.shape
    _, G, Tc, _ = k_cmp.shape
    assert HQ % G == 0
    QGS = HQ // G

    if scale is None:
        scale = D ** -0.5

    # Expand k_cmp, v_cmp to match query heads
    k_exp = k_cmp.unsqueeze(2).expand(-1, -1, QGS, -1, -1).reshape(B, HQ, Tc, D)
    v_exp = v_cmp.unsqueeze(2).expand(-1, -1, QGS, -1, -1).reshape(B, HQ, Tc, D)

    # Compute in float32 for stability
    q_f = q.float()
    k_f = k_exp.float()
    v_f = v_exp.float()

    # Attention scores: [B, HQ, S, Tc]
    scores = torch.einsum("bhsd,bhtd->bhst", q_f, k_f) * scale

    # Block-level causal mask
    query_pos = torch.arange(S, device=q.device).view(1, 1, S, 1)
    block_idx = torch.arange(Tc, device=q.device).view(1, 1, 1, Tc)
    causal_mask = block_idx < (query_pos + 1) // block_size
    scores = scores.masked_fill(~causal_mask, float("-inf"))

    # LSE and softmax
    lse_cmp = torch.logsumexp(scores, dim=-1)  # [B, HQ, S]
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    o_cmp = torch.einsum("bhst,bhtd->bhsd", attn, v_f)
    return o_cmp.to(q.dtype), lse_cmp


def topk_block_selection_ref(q, k_cmp, lse_cmp, block_size, block_count, scale=None):
    """
    Top-K block selection based on importance scores.

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

    # Importance = sum over query heads of exp(scores - lse)
    lse_grouped = lse_cmp.reshape(B, G, QGS, S).unsqueeze(-1)  # [B, G, QGS, S, 1]
    importance = torch.exp(scores - lse_grouped)
    importance = importance.sum(dim=2)  # [B, G, S, Tc]

    # Block-level causal mask
    query_pos = torch.arange(S, device=q.device).view(1, 1, S, 1)
    block_idx = torch.arange(Tc, device=q.device).view(1, 1, 1, Tc)
    causal_mask = block_idx < (query_pos + 1) // block_size
    importance = importance.masked_fill(~causal_mask, float("-inf"))

    # Top-K
    actual_k = min(block_count, Tc)
    _, indices = torch.topk(importance, k=actual_k, dim=-1)  # [B, G, S, actual_k]

    # Pad if fewer blocks available than requested
    if actual_k < block_count:
        pad_indices = torch.zeros(
            B, G, S, block_count - actual_k,
            dtype=indices.dtype, device=indices.device,
        )
        indices = torch.cat([indices, pad_indices], dim=-1)

    return indices


def topk_from_importance_ref(importance, block_size, block_count, num_init=0, num_local=0):
    """
    Top-K block selection from pre-computed importance scores.

    Standalone reference for the @dispatch("topk_block_selection") op.
    Supports reserved initial and local blocks (matching Scalable-Flash-NSA).

    Args:
        importance: [B, G, S, Tc] — pre-computed importance scores
        block_size: int — block size for causal boundary computation
        block_count: int — number of blocks to select
        num_init: int — number of initial blocks to always include (default: 0)
        num_local: int — number of local (nearest) blocks to always include (default: 0)

    Returns:
        block_indices: [B, G, S, block_count]
    """
    B, G, S, Tc = importance.shape
    device = importance.device

    importance = importance.float().clone()

    # Block-level causal mask
    query_pos = torch.arange(S, device=device).view(1, 1, S, 1)
    block_idx = torch.arange(Tc, device=device).view(1, 1, 1, Tc)
    causal_mask = block_idx < (query_pos + 1) // block_size
    importance = importance.masked_fill(~causal_mask, float("-inf"))

    # Output indices
    indices = torch.zeros(B, G, S, block_count, dtype=torch.long, device=device)

    for b in range(B):
        for g in range(G):
            for s in range(S):
                imp = importance[b, g, s].clone()  # [Tc]
                slot = 0

                # Phase 1: Force-include initial blocks
                for i in range(num_init):
                    if slot >= block_count:
                        break
                    if i < Tc:
                        indices[b, g, s, slot] = i
                        imp[i] = float("-inf")  # exclude from further selection
                        slot += 1

                # Phase 2: Force-include local (nearest) blocks
                current_block = s // block_size
                for i in range(num_local):
                    if slot >= block_count:
                        break
                    local_idx = current_block - i
                    if 0 <= local_idx < Tc and local_idx < (s + 1) // block_size:
                        indices[b, g, s, slot] = local_idx
                        imp[local_idx] = float("-inf")
                        slot += 1

                # Phase 3: Iterative argmax for remaining slots
                while slot < block_count:
                    if imp.max() == float("-inf"):
                        break  # no more valid blocks
                    idx = imp.argmax()
                    indices[b, g, s, slot] = idx
                    imp[idx] = float("-inf")
                    slot += 1

    return indices


def selection_attention_ref(q, k, v, block_indices, block_size, scale=None):
    """
    Selection attention: attend to selected K/V blocks per query position.

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
    B, HQ, S, D = q.shape
    _, G, _, _ = k.shape
    QGS = HQ // G
    block_count = block_indices.shape[-1]
    BS = block_size

    if scale is None:
        scale = D ** -0.5

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    o_slc = torch.zeros(B, HQ, S, D, device=q.device, dtype=torch.float32)

    for b in range(B):
        for g in range(G):
            for t in range(S):
                # block indices for this position: [block_count]
                bi = block_indices[b, g, t]

                # Expand to individual positions: [block_count * BS]
                kv_indices = (bi.unsqueeze(-1) * BS + torch.arange(BS, device=q.device)).reshape(-1)
                kv_indices = kv_indices.clamp(0, S - 1)

                # Gather K/V: [block_count * BS, D]
                k_sel = k_f[b, g, kv_indices]
                v_sel = v_f[b, g, kv_indices]

                for qh in range(QGS):
                    h = g * QGS + qh
                    q_t = q_f[b, h, t] * scale  # [D]

                    # Scores: [block_count * BS]
                    attn = torch.einsum("d,nd->n", q_t, k_sel)

                    # Causal mask
                    attn = attn.masked_fill(kv_indices > t, float("-inf"))

                    attn = torch.softmax(attn, dim=0)
                    attn = torch.nan_to_num(attn, nan=0.0)

                    o_slc[b, h, t] = torch.einsum("n,nd->d", attn, v_sel)

    return o_slc.to(q.dtype)


def sliding_window_attention_ref(q, k, v, window_size, scale=None):
    """
    Sliding window causal attention.

    Each query at position t attends to positions [max(0, t-W+1), t]
    where W = window_size.

    Args:
        q: [B, HQ, S, D]
        k: [B, G, S, D]
        v: [B, G, S, D]
        window_size: int
        scale: float

    Returns:
        o_swa: [B, HQ, S, D]
    """
    B, HQ, S, D = q.shape
    _, G, _, _ = k.shape
    QGS = HQ // G

    if scale is None:
        scale = D ** -0.5

    # Expand k, v to match query heads
    k_exp = k.unsqueeze(2).expand(-1, -1, QGS, -1, -1).reshape(B, HQ, S, D)
    v_exp = v.unsqueeze(2).expand(-1, -1, QGS, -1, -1).reshape(B, HQ, S, D)

    q_f = q.float()
    k_f = k_exp.float()
    v_f = v_exp.float()

    # Scores: [B, HQ, S, S]
    scores = torch.einsum("bhsd,bhtd->bhst", q_f, k_f) * scale

    # Combined causal + window mask
    query_pos = torch.arange(S, device=q.device).view(1, 1, S, 1)
    key_pos = torch.arange(S, device=q.device).view(1, 1, 1, S)
    mask = (key_pos <= query_pos) & (key_pos >= query_pos - window_size + 1)
    scores = scores.masked_fill(~mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    o_swa = torch.einsum("bhst,bhtd->bhsd", attn, v_f)
    return o_swa.to(q.dtype)


def nsa_forward_ref(q, k, v, g_cmp, g_slc, g_swa,
                    block_size, block_count, window_size, scale=None):
    """
    Full NSA forward pass reference.

    Args:
        q: [B, HQ, S, D]
        k: [B, G, S, D]
        v: [B, G, S, D]
        g_cmp: [B, HQ, S] — compression gate
        g_slc: [B, HQ, S] — selection gate
        g_swa: [B, HQ, S] — sliding window gate
        block_size: int
        block_count: int
        window_size: int
        scale: float

    Returns:
        o: [B, HQ, S, D]
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Stage 1: Mean Pool
    k_cmp, v_cmp = mean_pool_kv(k, v, block_size)

    # Stage 2: Compression Attention
    o_cmp, lse_cmp = compression_attention_ref(q, k_cmp, v_cmp, block_size, scale)

    # Stage 3: Top-K Selection
    block_indices = topk_block_selection_ref(q, k_cmp, lse_cmp, block_size, block_count, scale)

    # Stage 4: Selection Attention
    o_slc = selection_attention_ref(q, k, v, block_indices, block_size, scale)

    # Stage 5: Sliding Window Attention
    o_swa = sliding_window_attention_ref(q, k, v, window_size, scale)

    # Stage 6: Gate Fusion
    g_cmp_s = torch.sigmoid(g_cmp.float()).unsqueeze(-1)  # [B, HQ, S, 1]
    g_slc_s = torch.sigmoid(g_slc.float()).unsqueeze(-1)
    g_swa_s = torch.sigmoid(g_swa.float()).unsqueeze(-1)

    o = g_cmp_s * o_cmp.float() + g_slc_s * o_slc.float() + g_swa_s * o_swa.float()
    return o.to(q.dtype)
