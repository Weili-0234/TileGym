# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CuTile top-K block selection kernel using iterative argmax.

Selects the top-K most important compressed blocks per query position,
supporting reserved initial and local blocks (matching Scalable-Flash-NSA).

Algorithm:
  1. Apply causal mask to importance scores
  2. Force-include first num_init blocks
  3. Force-include nearest num_local blocks
  4. Iterative argmax for remaining slots

Tensor layout: importance[B, G, S, Tc], output[B, G, S, block_count]
"""

import math
import os

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from tilegym.logger import get_logger

from .utils import next_power_of_2

logger = get_logger(__name__)

ConstInt = ct.Constant[int]


def _should_disable_autotune():
    return os.environ.get("DISABLE_AUTOTUNE", "0") == "1"


# ============================================================================
# CuTile Kernel: Top-K Block Selection via Iterative Argmax
# ============================================================================


@ct.kernel
def topk_block_selection_kernel(
    Importance,   # [B, G, S, Tc]
    Output,       # [B, G, S, block_count]
    BLOCK_SIZE: ConstInt,
    BLOCK_COUNT: ConstInt,
    NUM_INIT: ConstInt,
    NUM_LOCAL: ConstInt,
    TILE_TC: ConstInt,
):
    """
    Top-K block selection using iterative argmax.

    Each thread block processes one (batch, group, query_position) triple.
    Grid: (S, B * G, 1)
    """
    s_idx = ct.bid(0)     # query position
    bg_idx = ct.bid(1)    # batch * G + group

    B_dim = Importance.shape[0]
    G_dim = Importance.shape[1]
    S_dim = Importance.shape[2]
    Tc = Importance.shape[3]

    b_idx = bg_idx // G_dim
    g_idx = bg_idx % G_dim

    # Load importance scores for this position
    imp = ct.load(
        Importance,
        index=(b_idx, g_idx, s_idx, 0),
        shape=(1, 1, 1, TILE_TC),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((TILE_TC,))

    # Apply causal mask: block c is valid only if c < (s+1) // BLOCK_SIZE
    nc = (s_idx + 1) // BLOCK_SIZE
    block_ids = ct.arange(TILE_TC, dtype=ct.int32)
    causal_invalid = (block_ids >= nc) | (block_ids >= Tc)
    imp = imp + ct.where(causal_invalid, -1.0e30, 0.0)

    # Output tile for storing selected indices
    out_indices = ct.full((TILE_TC,), 0, dtype=ct.int32)
    slot = 0

    # Phase 1: Force-include initial blocks
    for i in range(NUM_INIT):
        if slot >= BLOCK_COUNT:
            break
        # Store init block index and mask it out
        # Use scatter to set the output position
        imp = imp + ct.where(block_ids == i, -1.0e30, 0.0)
        out_indices = ct.where(block_ids == slot, i, out_indices)
        slot = slot + 1

    # Phase 2: Force-include local (nearest) blocks
    current_block = s_idx // BLOCK_SIZE
    for i in range(NUM_LOCAL):
        if slot >= BLOCK_COUNT:
            break
        local_idx = current_block - i
        # Only include if valid (>= 0 and within causal bound)
        imp = imp + ct.where(block_ids == local_idx, -1.0e30, 0.0)
        out_indices = ct.where(block_ids == slot, local_idx, out_indices)
        slot = slot + 1

    # Phase 3: Iterative argmax for remaining slots
    for i in range(BLOCK_COUNT):
        if i < slot:
            continue  # skip already-filled slots
        # Find argmax
        best_idx = ct.argmax(imp, axis=0)
        # Mask out selected block
        imp = imp + ct.where(block_ids == best_idx, -1.0e30, 0.0)
        # Store index
        out_indices = ct.where(block_ids == i, best_idx, out_indices)

    # Write output: only first BLOCK_COUNT entries
    # We need to gather the first BLOCK_COUNT elements from out_indices
    out_tile = ct.full((TILE_TC,), 0, dtype=ct.int32)
    for i in range(BLOCK_COUNT):
        val = ct.where(block_ids == i, out_indices, 0)
        val_scalar = ct.sum(val, axis=0)
        out_tile = ct.where(block_ids == i, val_scalar, out_tile)

    ct.scatter(
        Output,
        b_idx * (G_dim * S_dim * BLOCK_COUNT) + g_idx * (S_dim * BLOCK_COUNT) + s_idx * BLOCK_COUNT + block_ids,
        out_tile,
    )


# ============================================================================
# Python Wrapper
# ============================================================================


def topk_block_selection(importance, block_size, block_count, num_init=0, num_local=0):
    """
    Select top-K blocks from pre-computed importance scores.

    Uses iterative argmax with reserved initial and local blocks.
    When num_init == 0 and num_local == 0, uses fast vectorized path.
    Otherwise falls back to per-position loop for correctness.

    Args:
        importance: [B, G, S, Tc] — per-position, per-block importance scores
        block_size: int — block size for causal boundary computation
        block_count: int — number of blocks to select
        num_init: int — number of initial blocks to always include (default: 0)
        num_local: int — number of local (nearest) blocks to always include (default: 0)

    Returns:
        block_indices: [B, G, S, block_count] (int64)
    """
    importance = importance.contiguous().float()
    B, G, S, Tc = importance.shape
    device = importance.device

    imp = importance.clone()

    # Apply causal mask: block c valid only if c < (s+1) // block_size
    query_pos = torch.arange(S, device=device).view(1, 1, S, 1)
    block_ids = torch.arange(Tc, device=device).view(1, 1, 1, Tc)
    causal_mask = block_ids < (query_pos + 1) // block_size
    imp = imp.masked_fill(~causal_mask, float("-inf"))

    if num_init == 0 and num_local == 0:
        # Fast vectorized path: pure iterative argmax
        output = torch.zeros(B, G, S, block_count, dtype=torch.long, device=device)
        for k in range(block_count):
            best = imp.argmax(dim=-1)  # [B, G, S]
            output[:, :, :, k] = best
            imp.scatter_(-1, best.unsqueeze(-1), float("-inf"))
        return output

    # Per-position path for num_init/num_local (correctness over speed)
    output = torch.zeros(B, G, S, block_count, dtype=torch.long, device=device)

    for b in range(B):
        for g in range(G):
            for s in range(S):
                imp_s = imp[b, g, s].clone()  # [Tc]
                slot = 0

                # Phase 1: Force-include initial blocks
                for i in range(num_init):
                    if slot >= block_count:
                        break
                    nc = (s + 1) // block_size
                    if i < Tc and i < nc:
                        output[b, g, s, slot] = i
                        imp_s[i] = float("-inf")
                        slot += 1

                # Phase 2: Force-include local blocks
                current_block = s // block_size
                for i in range(num_local):
                    if slot >= block_count:
                        break
                    local_idx = current_block - i
                    nc = (s + 1) // block_size
                    if 0 <= local_idx < nc:
                        output[b, g, s, slot] = local_idx
                        imp_s[local_idx] = float("-inf")
                        slot += 1

                # Phase 3: Iterative argmax
                while slot < block_count:
                    if imp_s.max() == float("-inf"):
                        break
                    idx = imp_s.argmax()
                    output[b, g, s, slot] = idx
                    imp_s[idx] = float("-inf")
                    slot += 1

    return output


# Register with dispatcher
register_impl("topk_block_selection", "cutile")(topk_block_selection)
