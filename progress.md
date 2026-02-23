# NSA Implementation — Progress Log

## Session 1

### Phase 1: Reference Implementation — COMPLETED
- Read FLA naive.py, TileLang reference.py
- Read TileGym attention.py, attention_sink.py patterns
- Created `nsa_reference.py` with 6 functions
- Created `test_nsa.py` with Layer 0 cross-validation
- Fixed imports (importlib for FLA/TileLang to bypass __init__)
- Fixed TileLang gate handling (raw vs sigmoid)
- **24/24 Layer 0 tests PASSED**

### Phase 2: Compression Attention Kernel — COMPLETED
- Adapted fmha_fwd_kernel_with_lse → nsa_compression_attn_kernel
- Key change: block-level causal mask `c < (t+1) // block_size`
- Hit NaN issue: first block_size-1 positions have NC=0 → all-masked → NaN
- Fix 1: nan_to_num in wrapper (worked but fragile)
- Fix 2: switched to -1e6 masking (attention_sink pattern) + zero-out early positions
- **12/12 compression attention tests PASSED**

### Phase 3: Sliding Window Kernel — COMPLETED
- Adapted attention_sink BANDWIDTH logic → nsa_sliding_window_kernel
- Key change: no sink tokens, window constraint from query position
- Hit NaN issue: KV blocks outside window → all-masked → NaN in online softmax
- Same -1e6 fix resolved it
- **12/12 sliding window tests PASSED**

### Phase 4: Top-K Selection — COMPLETED
- Implemented compute_importance_and_select in nsa.py
- Pure PyTorch: importance = sum(exp(q·k_cmp - lse)) + topk
- Tested via selection attention integration

### Phase 5: Selection Attention — COMPLETED (PyTorch fallback)
- Uses selection_attention_ref (PyTorch) as fallback
- Data-dependent ct.load not explored — deferred
- **16/16 selection attention tests PASSED**

### Phase 6: Integration — COMPLETED
- tile_nsa wrapper written, register_impl done
- Added `from . import nsa` to `cutile/__init__.py`
- Added `@dispatch("nsa")` interface to `ops.py`
- Enabled E2E tests in `TestNSAEndToEnd`
- **4/4 E2E tests PASSED**

### Total: 68/68 all tests passed
- Layer 0 (reference cross-validation): 24 tests
- Layer 1 (kernel unit tests): 40 tests (compression: 12, sliding window: 12, selection: 16)
- Layer 2 (end-to-end): 4 tests

---

## Session 2: Cross-Validation Research & Test Expansion Planning

### Research Completed
1. **nsa-impl** (`/root/workspace/nsa-impl/`): Full 3-branch NSA implementation
   - Compression: `flex_attention` + `@torch.compile` + block mask
   - Selection: Custom Triton kernel (one-pass) + FLA ParallelNSAFunction (two-pass)
   - Sliding window: `flash_attn_func`
   - Gate fusion: direct multipliers (no sigmoid)

2. **vLLM** (`/root/workspace/vllm/`): Token-level sparse MLA, NOT NSA
   - Completely different architecture (no compression/sliding/gates)
   - Cannot be used for cross-validation

3. **TileLang** (`/root/workspace/tilelang/examples/deepseek_nsa/`): Selection + sliding only
   - Confirmed: NO compression branch

### Critical Discovery: Compression Causal Mask Conventions
Three different conventions exist (detailed in findings.md #12):
- **Convention A** (FLA / ours): strict past-only, `c < (t+1)//BS`
- **Convention B** (hypothetical): block-causal + intra-block bi-directional, `c <= t//BS`
- **Convention C** (nsa-impl actual): block-anti-causal, `c >= t//BS` — sees current+future, drops past!
- nsa-impl's mask is NOT "block-causal + intra-bi" — it's the opposite direction
- Consequence: compression attention and E2E cannot be cross-validated against nsa-impl

### Test Expansion Plan Drafted
- Target: 68 → 106 tests (+38 new)
- New cross-validation: selection attention vs nsa-impl one-pass Triton kernel
- Edge cases: irregular S, large GQA, boundary conditions
- E2E: add fp16, B>1, new shapes
- Plan file at `/root/.claude/plans/serene-crafting-swan.md`

### Status: Plan drafted, pending user approval to implement
