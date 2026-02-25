"""Reproduce the report figures.

This script generates:
  - best_footpath.png
  - constrained_footpath.png
  - transmission_angle.png

Run from the code/ folder:

    python make_figures.py

It writes images into the parent folder (the LaTeX project folder).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from linkage import LinkageParams, point_C, transmission_angle_mu
from analysis_tools import stance_metrics


ROOT = Path(__file__).resolve().parents[1]


def plot_footpath(
    p: LinkageParams,
    out_name: str,
    title: str,
    *,
    n_phi: int = 3000,
) -> None:
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    C = point_C(phis, p, branch=+1)
    x = C[:, 0]
    y = C[:, 1]

    idx, _ = stance_metrics(x, y)
    xs = x[idx]
    ys = y[idx]

    X = np.vstack([xs, np.ones_like(xs)]).T
    m, q = np.linalg.lstsq(X, ys, rcond=None)[0]

    xx = np.linspace(xs.min(), xs.max(), 200)
    yy = m * xx + q

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label="full trajectory")
    plt.plot(xs, ys, linewidth=4, label="stance (near-straight)")
    plt.plot(xx, yy, linestyle="--", linewidth=2.5, label="best-fit line")

    plt.xlabel("x / l")
    plt.ylabel("y / l")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(ROOT / out_name, dpi=1200)
    plt.close()


def plot_transmission_compare(pA: LinkageParams, pB: LinkageParams, out_name: str, *, n_phi: int = 3000) -> None:
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    muA = transmission_angle_mu(phis, pA)
    muB = transmission_angle_mu(phis, pB)

    plt.figure(figsize=(6, 4))
    plt.plot(np.degrees(phis), muA, label="Design A (best straightness)")
    plt.plot(np.degrees(phis), muB, label="Design B (min μ ≥ 30°)")
    plt.axhline(30.0, linestyle="--", linewidth=1.5, label="30° threshold")

    plt.xlabel("input angle φ (deg)")
    plt.ylabel("transmission angle μ (deg)")
    plt.ylim(0, 180)

    plt.legend()
    plt.tight_layout()

    plt.savefig(ROOT / out_name, dpi=1200)
    plt.close()


def main() -> None:
    # Design A (path-based)
    pA = LinkageParams(a=1.13, b=1.32, c=1.20, d=1.46, l=1.0)

    # Design B (path-based + transmission-angle constraint)
    pB = LinkageParams(a=1.97, b=1.92, c=1.52, d=2.26, l=1.0)

    plot_footpath(pA, "best_footpath.pdf", "Optimised foot trajectory (Design A)")
    plot_footpath(pB, "constrained_footpath.pdf", "Foot trajectory with transmission-angle constraint")

    plot_transmission_compare(pA, pB, "transmission_angle.pdf")

    print("Wrote figures to:", ROOT)


if __name__ == "__main__":
    main()
