# NSA Implementation — Technical Findings

## 1. Causal Condition for Compression Attention
- From FLA parallel.py: `NC = (i_t + 1) // BS`
- Query at position t can attend to compressed block c if `c < (t+1) // block_size`
- First block_size-1 positions attend to NO compressed blocks (NC=0)
- Position block_size-1 can attend to block 0 (its own block's mean)

## 2. Layout Convention
- TileGym: `[B, H, S, D]` (head-first)
- FLA/TileLang: `[B, T, H, D]` (seq-first)
- Cross-validation requires transpose between dim 1 and 2

## 3. FLA naive_nsa API
- Only implements selection attention (takes pre-computed block_indices)
- block_indices: [B, T, H, S] per KV-head, expanded to all query heads internally
- Causal mask: `i_i > i_q` (gathered kv position > query position)
- No gates in FLA naive

## 4. TileLang naive_nsa API
- Combines selection + sliding window with g_slc and g_swa gates
- **Gates are raw multipliers** (NOT sigmoid-activated). Cross-validation must pass post-sigmoid values
- No compression branch
- Same layout as FLA (seq-first)

## 5. Mean Pooling
- Partial last blocks must be handled correctly (mean over actual positions, not zero-padded)
- Efficient path for S % block_size == 0 uses reshape+mean; fallback uses Python loop

## 6. FLA/TileLang Import Pattern
- **Cannot** import `fla.ops.nsa.naive` normally because `fla.__init__` imports all layers which require `einops`
- **Solution**: Use `importlib.util.spec_from_file_location` to load the file directly, bypassing `__init__.py`
- Same approach needed for tilelang reference

---

## CRITICAL: Online Softmax NaN Issue (the -inf vs -1e6 problem)

### Problem
When ALL positions in a KV tile block are masked (e.g., first block_size-1 query positions in compression attention, or KV blocks outside the sliding window), using `-math.inf` for masking produces NaN:

```
qk = [original_values] + (-inf) = [-inf, -inf, ...]
max(qk) = -inf
m_ij = max(m_i, -inf * qk_scale) = -inf
qk * qk_scale - m_ij = (-inf) - (-inf) = NaN  ← IEEE 754: inf - inf = NaN
p = exp2(NaN) = NaN  → corrupts all subsequent accumulations
```

This **never** happens in standard FMHA because every query has at least one valid KV position (itself). But in NSA:
- **Compression attention**: First block_size-1 positions have NC=0 (no valid compressed blocks)
- **Sliding window**: KV blocks can fall entirely outside the window for certain query positions

### Solution: Use -1e6 (attention_sink.py pattern)
Replace `-math.inf` with `-1.0e6` in all mask computations:
```python
# WRONG: causes NaN
mask = ct.where(valid_mask, 0.0, -math.inf)

# CORRECT: -1e6 avoids NaN, flush_to_zero handles the rest
invalid = oob | not_causal | too_old
qk = qk + ct.where(invalid, -1.0e6, 0.0)
```

### Why -1e6 Works
When ALL entries are -1e6:
- `max(qk) = -1e6`, `m_ij = -1e6 * scale ≈ -180000`
- `qk * scale - m_ij = 0`, `p = exp2(0) = 1`, `l_ij = TILE_N`
- This is "wrong" (gives non-zero output for masked positions)

But when the NEXT valid block arrives:
- `m_ij_new ≈ 0.9` (from valid entries)
- `alpha = exp2(-180000 - 0.9) ≈ 0` (flush_to_zero!)
- `l_i = TILE_N * 0 + l_ij_new = l_ij_new` (previous contribution wiped out)

The online softmax's rescaling mechanism naturally erases the incorrect contribution from all-masked blocks, as long as `flush_to_zero=True` is used in `ct.exp2`.

### Compression Attention: Zero-Out Early Positions
Even with -1e6, positions 0 to block_size-2 produce small non-zero artifacts (uniform softmax over -1e6 values). Since these positions semantically should have zero compression output, the Python wrapper explicitly zeros them:
```python
if block_size > 1:
    o[:, :, :block_size - 1] = 0
```

---

## 7. Selection Attention: PyTorch Fallback
- Selection attention requires data-dependent `ct.load` (indirect indexing by `block_indices`)
- This is NOT verified in cutile — deferred to future optimization
- Current implementation uses the pure PyTorch reference (`selection_attention_ref`)
- This is legitimate: the wrapper still registers as `cutile` backend (other ops do this too)

## 8. K Loading Pattern (Transposed)
The standard cutile pattern for loading K transposed uses the `order` parameter:
```python
k = ct.load(K, index=(b, h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),  # swap dims 2,3 → [D, S] loaded as transposed
            latency=2)
```
- `order=(0,1,3,2)` logically transposes the last two dims
- Physical layout: K is [B, H, S, D], but we want [D, TILE_N] for Q @ K^T
- The `j` in dim 3 selects the j-th tile of size TILE_N in the S dimension
- The `0` in dim 2 always starts from 0 (loads all D elements)

## 9. Autotuning
- All kernels share the same tile config: `TILE_M ∈ {64,128,256}`, `TILE_N ∈ {32,64,128}`
- `DISABLE_AUTOTUNE=1` env var skips autotuning for CI (uses first config)
- `ct_experimental.autotune_launch` handles grid computation and config search

## 10. Test Tolerances
- bf16/fp16 kernels: `atol=5e-2, rtol=1e-2` (matches TileGym existing attention tests)
- Reference cross-validation (float32): `atol=1e-3, rtol=1e-3`

---

## 11. Reference Implementations Landscape (Session 2 Research)

### Available Cross-Validation Targets

| Repo | Location | Branches | Layout |
|------|----------|----------|--------|
| FLA naive | `/root/workspace/flash-linear-attention/fla/ops/nsa/naive.py` | Selection only | `[B, T, H, D]` seq-first |
| TileLang ref | `/root/workspace/tilelang/examples/deepseek_nsa/reference.py` | Selection + Sliding window | `[B, T, H, D]` seq-first |
| nsa-impl | `/root/workspace/nsa-impl/nsa/` | **All 3** (compression, selection, sliding) | `[B, M, H, D]` seq-first |
| vLLM sparse MLA | `/root/workspace/vllm/vllm/v1/attention/backends/mla/` | Token-level sparse (NOT NSA) | `[S, H, D]` no batch |

### Key: No third-party implementation uses the same compression causal mask as us (FLA convention)

---

## 12. CRITICAL: Compression Causal Mask — Three Conventions Compared

### Convention A: "Strict Past-Only" (FLA / Our Reference)
```python
# nsa_reference.py:94
causal_mask = block_idx < (query_pos + 1) // block_size
```
- Equivalent: `query_t >= (block_c + 1) * BS - 1`
- Meaning: Block must be **fully completed** before any query can see it
- Position 0: sees NO blocks (NC=0)
- Position BS-1: sees block 0 (first time)
- Position BS: sees block 0 (block 0 stays visible forever)
- **NaN risk**: positions 0 to BS-2 have NC=0 → all-masked → requires -1e6 fix

### Convention B: "Block-Causal + Intra-Block Bi-Directional" (hypothetical correct version)
```python
causal_mask = kv_idx <= q_idx // block_size
```
- Meaning: Query can see its own block AND all past blocks
- Position 0: sees block 0 (its own — includes "future" positions 1..BS-1 via mean)
- Position BS: sees blocks 0, 1
- **No NaN risk**: every position sees at least its own block
- **Mild information leakage**: within-block future positions visible through mean pool

### Convention C: "Block-Anti-Causal" (nsa-impl ACTUAL implementation)
```python
# nsa-impl nsa.py:40-41
return q_idx <= (kv_idx + 1) * block_size - 1
```
- Equivalent: `kv_idx >= floor(q_idx / BS)`
- Meaning: Query sees its own block AND all **future** blocks, but NOT past blocks
- Position 0: sees ALL blocks (0, 1, 2, ...)
- Position BS: sees blocks 1, 2, ... (block 0 DROPPED!)
- **No NaN risk**: always sees current + future blocks
- **Heavy information leakage**: can see all future compressed blocks
- **Past blocks lost**: unlike A and B, completed blocks become invisible

### Worked Example (BS=4, 2 blocks covering positions 0-7)

| Query Pos | Conv. A (ours) | Conv. B (causal+bi) | Conv. C (nsa-impl) |
|-----------|---------------|--------------------|--------------------|
| 0 | none | block 0 | block 0, 1 |
| 1 | none | block 0 | block 0, 1 |
| 2 | none | block 0 | block 0, 1 |
| 3 | block 0 | block 0 | block 0, 1 |
| 4 | block 0 | block 0, 1 | block 1 only |
| 5 | block 0 | block 0, 1 | block 1 only |
| 6 | block 0 | block 0, 1 | block 1 only |
| 7 | block 0, 1 | block 0, 1 | block 1 only |

### Implication for Cross-Validation
- Compression attention CANNOT be cross-validated between our ref and nsa-impl
- Top-K selection depends on compression LSE → also differs
- E2E outputs differ even with identical inputs
- Only selection attention (same causal: kv_pos <= query_pos) and sliding window can be cross-validated

---

## 13. nsa-impl API Details

### Tensor Shapes (all seq-first)
- q: `[B, M, H, D]`, k/v: `[B, N, G, D]`
- gates: `[B, M, H]` — applied as **direct multipliers** (no sigmoid inside nsa_func)
- block_indices: `[B, M, G, T]`

### Layout Transform (ours → nsa-impl)
```python
q_nsa = q.transpose(1, 2)           # [B,H,S,D] → [B,S,H,D]
k_nsa = k.transpose(1, 2)           # [B,G,S,D] → [B,S,G,D]
bi_nsa = bi.permute(0, 2, 1, 3)     # [B,G,S,T] → [B,S,G,T]
g_nsa = g.transpose(1, 2)           # [B,H,S]   → [B,S,H]
```

### Gate Handling Difference
- Our ref: gates are **pre-sigmoid** (raw logits), sigmoid applied inside `nsa_forward_ref`
- nsa-impl: gates are **post-sigmoid** or arbitrary, multiplied directly
- TileLang: same as nsa-impl (raw multipliers)
- Cross-validation must pass `torch.sigmoid(g)` to nsa-impl/TileLang

### Selection Attention Variants
- `variant='two-pass'` (default): delegates to FLA's `ParallelNSAFunction`
- `variant='one-pass'`: custom Triton kernel (`SelectionAttention` class)
- One-pass kernel: `_sel_attn_fwd_kernel` with autotune over num_warps {1,2,4,8}
- Causal condition (prefill M=N): `cols <= m` where m = query position — matches our ref

### Dependencies
- flash_attn (sliding window), fla (mean_pooling, parallel_nsa_topk, ParallelNSAFunction)
- torch.nn.attention.flex_attention (compression)
- No __init__.py — uses relative imports, needs importlib trick

---

## 14. vLLM Sparse MLA — NOT the Same as NSA

vLLM implements **token-level sparse MLA** for DeepSeek V3.2 inference, fundamentally different from NSA:
- **No compression branch** (no mean pooling, no compressed KV)
- **No sliding window branch** (as separate path)
- **No gate fusion** (single sparse attention path)
- **Token-level granularity** (topk≈2048 individual tokens, not blocks of 32/64)
- **FP8 quantization** integral to the pipeline
- **No explicit causal mask** — causality implicit in KV cache (only past tokens exist)
- Sparse backends: FlashMLA (H100/Blackwell), FlashInfer (Blackwell), ROCm AIter (MI300)
- Cannot be used as cross-validation target for our NSA implementation

---

## 15. Test Coverage Gaps Identified (Session 2)

### Current Coverage
- Layer 0: B=1, S≤256, D=64
- Layer 1: S∈{128,256,512}, D∈{64,128}, no irregular shapes
- Layer 2: 2 shapes, fixed block_count=4, window_size=64, bf16 only

### Missing Scenarios
- **Irregular S**: not divisible by block_size (e.g., S=96, 160)
- **Large GQA ratios**: HQ=64 G=1 (64:1)
- **Edge cases**: S==block_size, block_count>Tc, window_size>S
- **E2E gaps**: no fp16, no B>1, no single-head (HQ=1,G=1)
