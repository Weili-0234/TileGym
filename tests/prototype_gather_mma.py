# Prototype: test ct.gather → ct.mma feasibility
# CuTile requires source files (uses inspect.getsourcelines)

import cuda.tile as ct
import torch
import math

ConstInt = ct.Constant[int]


@ct.kernel
def gather_mma_test_kernel(A, B, Indices, Output, TILE_M: ConstInt, TILE_K: ConstInt, TILE_N: ConstInt):
    """Test: gather rows of A using Indices, then matmul with B."""
    bid = ct.bid(0)

    # Load indices (data-dependent row selection)
    idx_tile = ct.arange(TILE_M, dtype=ct.int32)
    row_indices = ct.gather(Indices, bid * TILE_M + idx_tile)  # [TILE_M] int indices

    # Gather rows from A using data-dependent indices
    col_indices = ct.arange(TILE_K, dtype=ct.int32)
    a_gathered = ct.gather(A, (row_indices[:, None], col_indices[None, :]))  # [TILE_M, TILE_K]

    # Load B tile normally
    b_tile = ct.load(B, index=(0, bid), shape=(TILE_K, TILE_N), padding_mode=ct.PaddingMode.ZERO)

    # MMA: gathered A @ B
    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
    acc = ct.mma(a_gathered, b_tile, acc)

    # Store result
    ct.store(Output, index=(bid, 0), tile=acc.astype(Output.dtype))


def test_gather_mma():
    M, K, N = 64, 64, 64
    TILE_M, TILE_K, TILE_N = 64, 64, 64

    torch.manual_seed(42)
    A = torch.randn(128, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, 128, (M,), device="cuda", dtype=torch.int32)
    output = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)

    grid = (1, 1, 1)
    ct.launch(
        torch.cuda.current_stream(), grid, gather_mma_test_kernel,
        (A, B, indices, output, TILE_M, TILE_K, TILE_N),
    )
    torch.cuda.synchronize()

    # Reference
    A_gathered_ref = A[indices.long()]  # [M, K]
    ref = A_gathered_ref.float() @ B.float()

    print(f"Output shape: {output.shape}")
    print(f"Max abs diff: {(output.float() - ref).abs().max().item():.6f}")
    print(f"Mean abs diff: {(output.float() - ref).abs().mean().item():.6f}")

    try:
        torch.testing.assert_close(output.float(), ref.float(), atol=1.0, rtol=0.05)
        print("PASS: ct.gather -> ct.mma works!")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


if __name__ == "__main__":
    success = test_gather_mma()
    exit(0 if success else 1)
