"""Random-search optimisation for the four-bar walking-foot linkage.

Usage (from the code/ folder):

    python optimize.py --samples 20000 --min-mu 0
    python optimize.py --samples 80000 --min-mu 30

This prints the best design found and its metrics.

NOTE: This is intentionally simple and transparent (not a heavy optimisation
library). If you want deterministic results, keep the seed fixed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from linkage import LinkageParams
from analysis_tools import evaluate_design


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-mu", type=float, default=0.0, help="minimum transmission angle in degrees")
    ap.add_argument("--min-duty", type=float, default=0.30)

    # Search ranges (dimensionless ratios because l defaults to 1)
    ap.add_argument("--a-range", type=float, nargs=2, default=(0.4, 2.0))
    ap.add_argument("--b-range", type=float, nargs=2, default=(0.4, 2.2))
    ap.add_argument("--c-range", type=float, nargs=2, default=(0.4, 2.2))
    ap.add_argument("--d-range", type=float, nargs=2, default=(0.1, 2.6))

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    best = None

    for _ in range(args.samples):
        a = rng.uniform(*args.a_range)
        b = rng.uniform(*args.b_range)
        c = rng.uniform(*args.c_range)
        d = rng.uniform(*args.d_range)

        p = LinkageParams(a=a, b=b, c=c, d=d, l=1.0)

        res = evaluate_design(p, min_mu_deg=args.min_mu, min_duty=args.min_duty)
        if res is None:
            continue

        if best is None or res.score > best.score:
            best = res

    if best is None:
        print("No feasible design found under the given constraints.")
        return

    p = best.params

    print("Best design found:")
    print(f"  a/l = {p.a:.4f}")
    print(f"  b/l = {p.b:.4f}")
    print(f"  c/l = {p.c:.4f}")
    print(f"  d/l = {p.d:.4f}")
    print()
    print("Metrics:")
    print(f"  score       = {best.score:.4f}")
    print(f"  duty factor = {best.duty:.4f}")
    print(f"  stride dx   = {best.dx:.4f}")
    print(f"  straight E  = {best.E:.4f}")
    print(f"  swing height= {best.swing_height:.4f}")
    print(f"  min mu (deg)= {best.min_mu:.4f}")


if __name__ == "__main__":
    main()
