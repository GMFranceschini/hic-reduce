"""
hic_reduce.py
-------------
Reduce a Hi-C matrix (.cool format) to its rank-n approximation by retaining
only the top n eigenvectors (truncated eigendecomposition).

Defaults: n=10, partial decomposition via ARPACK (fast, sparse-aware).
If --auto is requested, the full spectrum is computed and the Wigner semicircle
law is fit to determine n automatically.

Two output modes (--output-mode):

  oe   (default): reconstruct in OE space.
                  M_out = Σ_k λ_k · v_k · v_k^T
                  Values are OE-scale floats. Distance decay removed.

  raw : project the original balanced matrix through the signal subspace.
                  P = V_n @ V_n^T
                  M_out = P @ M_raw @ P
                  Values in original count scale. Distance decay preserved.

References:
  Franzini, Di Stefano & Micheletti (2021) essHi-C. https://arxiv.org/abs/2101.10645

Usage (CLI):
  python hic_reduce.py input.cool output.cool                        # n=10, oe, fast
  python hic_reduce.py input.cool output.cool --n-components 20
  python hic_reduce.py input.cool output.cool --auto                 # full eigh + Wigner fit
  python hic_reduce.py input.cool output.cool --output-mode raw
  python hic_reduce.py input.cool output.cool --resolution 10000     # from .mcool

Usage (API):
  from hic_reduce import load_chrom_matrix, oe_normalize
  from hic_reduce import partial_eigenvectors, full_eigenvectors, wigner_cutoff
  from hic_reduce import reconstruct_oe, reconstruct_raw
"""

import argparse
import logging

import cooler
import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def oe_normalize(M: np.ndarray) -> np.ndarray:
    """OE normalisation by diagonal (distance) averaging. Zeros treated as missing."""
    N = M.shape[0]
    oe = np.zeros_like(M, dtype=float)
    for d in range(N):
        idx = np.arange(N - d)
        vals = M[idx, idx + d]
        valid = vals[vals > 0]
        if valid.size == 0:
            continue
        mean = valid.mean()
        oe[idx, idx + d] = vals / mean
        if d > 0:
            oe[idx + d, idx] = vals / mean
    return oe


def partial_eigenvectors(M: np.ndarray, n: int):
    """
    Compute only the top-n eigenpairs via ARPACK (fast path).
    M should be symmetric; can be dense or sparse.
    Returns (eigenvalues, eigenvectors) sorted by |λ| descending.
    """
    N = M.shape[0]
    # eigsh requires k < N - 1
    k = min(n, N - 2)
    eigenvalues, eigenvectors = eigsh(M, k=k, which="LM")
    order = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def full_eigenvectors(M: np.ndarray):
    """
    Full eigendecomposition via numpy (slow path, needed for Wigner fit).
    Returns (eigenvalues, eigenvectors) sorted by |λ| descending.
    """
    M = (M + M.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def wigner_cutoff(eigenvalues: np.ndarray, n_bins: int = 50):
    """
    Estimate the number of essential components using the Wigner semicircle law.

        p(λ) = (2 / π Λ²) * sqrt(Λ² − λ²),   |λ| ≤ Λ

    Fit this to the full eigenvalue histogram; eigenspaces with |λ| > Λ are essential.

    Returns
    -------
    n_essential : int
    Lambda_fit  : float
    """
    def semicircle(lam, Lambda):
        inside = np.maximum(Lambda**2 - lam**2, 0.0)
        return (2.0 / (np.pi * Lambda**2)) * np.sqrt(inside)

    counts, edges = np.histogram(eigenvalues, bins=n_bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    lam0 = np.std(eigenvalues) * 2.0

    try:
        (Lambda_fit,), _ = curve_fit(
            semicircle, centres, counts,
            p0=[lam0], bounds=(0, np.inf), maxfev=5000,
        )
    except RuntimeError:
        log.warning("Wigner fit did not converge; falling back to Λ = 2·std(λ).")
        Lambda_fit = lam0

    n_essential = int(np.sum(np.abs(eigenvalues) > Lambda_fit))
    log.info(f"  Wigner semicircle fit: Λ = {Lambda_fit:.4f}  →  {n_essential} essential components")
    return n_essential, float(Lambda_fit)


def reconstruct_oe(eigenvalues: np.ndarray, eigenvectors: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct in OE space: M_out = Σ_k λ_k · v_k · v_k^T"""
    ev = eigenvalues[:n]
    V  = eigenvectors[:, :n]
    return (V * ev) @ V.T


def reconstruct_raw(M_raw: np.ndarray, eigenvectors: np.ndarray, n: int) -> np.ndarray:
    """
    Project original balanced matrix onto the signal subspace.
        P     = V_n @ V_n^T
        M_out = P @ M_raw @ P
    """
    V = eigenvectors[:, :n]
    P = V @ V.T
    return P @ M_raw @ P


# ---------------------------------------------------------------------------
# cooler I/O
# ---------------------------------------------------------------------------

def load_chrom_matrix(clr: cooler.Cooler, chrom: str, balance: bool = True) -> np.ndarray:
    M = clr.matrix(balance=balance).fetch(chrom).astype(float)
    return np.nan_to_num(M, nan=0.0)


def write_cool(output_path: str, chrom_matrices: dict, clr_source: cooler.Cooler) -> None:
    import pandas as pd

    bins = clr_source.bins()[:]
    pixels_list = []

    for chrom, M in chrom_matrices.items():
        chrom_bins = bins[bins["chrom"] == chrom]
        if chrom_bins.empty:
            continue
        bin_offset = chrom_bins.index[0]
        n = M.shape[0]

        rows, cols = np.triu_indices(n)
        values = M[rows, cols]
        mask = values != 0

        pixels_list.append(pd.DataFrame({
            "bin1_id": rows[mask] + bin_offset,
            "bin2_id": cols[mask] + bin_offset,
            "count":   values[mask],
        }))

    all_pixels = (
        pd.concat(pixels_list, ignore_index=True)
        .sort_values(["bin1_id", "bin2_id"])
    )

    cooler.create_cooler(
        output_path,
        bins=bins[["chrom", "start", "end"]],
        pixels=all_pixels,
        dtypes={"count": float},
        assembly=clr_source.info.get("genome-assembly", None),
    )
    log.info(f"Written: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Reduce Hi-C .cool matrix to its essential component reconstruction.\n"
            "Default: top 10 components, fast partial decomposition (ARPACK).\n"
            "Use --auto for Wigner semicircle cutoff (requires full decomposition)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input",  help="Input .cool or .mcool file")
    p.add_argument("output", help="Output .cool file")

    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--n-components", "-n", type=int, default=10,
        help="Number of eigenvectors to retain (default: 10, uses fast ARPACK decomposition)",
    )
    grp.add_argument(
        "--auto", action="store_true",
        help="Auto-select n via Wigner semicircle fit (requires full eigendecomposition — slow at high resolution)",
    )

    p.add_argument(
        "--output-mode", choices=["oe", "raw"], default="oe",
        help=(
            "'oe' (default): reconstruct in OE space (float, no distance decay). "
            "'raw': project original balanced matrix through signal subspace "
            "(count scale, distance decay preserved)."
        ),
    )
    p.add_argument("--chroms", nargs="+", default=None,
                   help="Chromosomes to process (default: all)")
    p.add_argument("--no-balance", action="store_true",
                   help="Skip ICE balancing weights")
    p.add_argument("--resolution", type=int, default=None,
                   help="Resolution to fetch from a .mcool (e.g. 10000)")
    return p.parse_args()


def main():
    args = parse_args()

    cool_path = args.input
    if args.resolution:
        cool_path = f"{args.input}::/resolutions/{args.resolution}"

    log.info(f"Opening: {cool_path}")
    clr = cooler.Cooler(cool_path)

    balance = not args.no_balance
    if balance and "weight" not in clr.bins().columns:
        log.warning("No 'weight' column found — falling back to raw counts.")
        balance = False

    chroms = args.chroms or clr.chromnames

    if args.auto:
        decomp_mode = "full (Wigner auto-cutoff)"
    else:
        decomp_mode = f"partial ARPACK, n={args.n_components}"

    log.info(f"Chromosomes  : {chroms}")
    log.info(f"Decomposition: {decomp_mode}")
    log.info(f"Output mode  : {args.output_mode}  |  balance={balance}")

    reduced = {}
    for chrom in chroms:
        log.info(f"  Processing {chrom} ...")

        M_raw = load_chrom_matrix(clr, chrom, balance=balance)
        M_oe  = oe_normalize(M_raw)

        if args.auto:
            eigenvalues, eigenvectors = full_eigenvectors(M_oe)
            n, _ = wigner_cutoff(eigenvalues)
            if n == 0:
                log.warning(f"  {chrom}: 0 essential components — skipping.")
                continue
        else:
            n = args.n_components
            eigenvalues, eigenvectors = partial_eigenvectors(M_oe, n)

        if args.output_mode == "oe":
            M_out = reconstruct_oe(eigenvalues, eigenvectors, n)
        else:
            M_out = reconstruct_raw(M_raw, eigenvectors, n)

        reduced[chrom] = M_out
        log.info(f"    {chrom}: {M_raw.shape[0]} bins → rank-{n} ({args.output_mode})")

    if not reduced:
        log.error("No chromosomes processed. Exiting.")
        return

    write_cool(args.output, reduced, clr_source=clr)


if __name__ == "__main__":
    main()
