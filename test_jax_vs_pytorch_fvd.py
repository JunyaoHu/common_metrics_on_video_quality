"""Test that JAX and PyTorch FVD implementations produce matching results."""

import sys
import numpy as np

def main():
    try:
        import torch
    except ImportError:
        print("SKIP: PyTorch not installed")
        return 0
    try:
        import jax
    except ImportError:
        print("SKIP: JAX not installed")
        return 0

    from calculate_fvd import calculate_fvd

    # Fixed seed for reproducibility
    rng = np.random.RandomState(42)

    B, T, C, H, W = 8, 16, 3, 64, 64
    data1 = rng.rand(B, T, C, H, W).astype(np.float32)
    data2 = rng.rand(B, T, C, H, W).astype(np.float32)

    videos1_pt = torch.from_numpy(data1)
    videos2_pt = torch.from_numpy(data2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- only_final=True comparison ---
    print("=" * 60)
    print("Test 1: only_final=True")
    print("=" * 60)

    result_pt = calculate_fvd(videos1_pt, videos2_pt, device,
                              method='videogpt', only_final=True)
    result_jax = calculate_fvd(videos1_pt, videos2_pt,
                               method='jax', only_final=True)

    val_pt = result_pt["value"][0]
    val_jax = result_jax["value"][0]
    rel_diff = abs(val_pt - val_jax) / max(abs(val_pt), 1e-8)

    print(f"  PyTorch (videogpt): {val_pt:.6f}")
    print(f"  JAX:                {val_jax:.6f}")
    print(f"  Relative diff:      {rel_diff:.6f} ({rel_diff*100:.4f}%)")

    final_pass = rel_diff < 0.01
    print(f"  Result:             {'PASS' if final_pass else 'FAIL'}")

    # --- only_final=False comparison ---
    print()
    print("=" * 60)
    print("Test 2: only_final=False (progressive)")
    print("=" * 60)

    result_pt_prog = calculate_fvd(videos1_pt, videos2_pt, device,
                                   method='videogpt', only_final=False)
    result_jax_prog = calculate_fvd(videos1_pt, videos2_pt,
                                    method='jax', only_final=False)

    vals_pt = result_pt_prog["value"]
    vals_jax = result_jax_prog["value"]

    assert len(vals_pt) == len(vals_jax), (
        f"Length mismatch: PyTorch={len(vals_pt)}, JAX={len(vals_jax)}")

    prog_pass = True
    print(f"  {'Frames':>8s}  {'PyTorch':>12s}  {'JAX':>12s}  {'RelDiff':>10s}  {'Status'}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*6}")

    for i, (vp, vj) in enumerate(zip(vals_pt, vals_jax)):
        n_frames = 10 + i
        rd = abs(vp - vj) / max(abs(vp), 1e-8)
        ok = rd < 0.01
        if not ok:
            prog_pass = False
        print(f"  {n_frames:>8d}  {vp:>12.4f}  {vj:>12.4f}  {rd:>10.6f}  {'PASS' if ok else 'FAIL'}")

    print(f"\n  Progressive result: {'PASS' if prog_pass else 'FAIL'}")

    # --- Summary ---
    print()
    print("=" * 60)
    all_pass = final_pass and prog_pass
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
