import numpy as np
import matplotlib.pyplot as plt

from linkage import LinkageParams, point_C


def _close_gaps_circular(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill short False-runs in a circular boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    if n == 0:
        return mask

    ext = np.concatenate([mask, mask, mask])
    start, end = n, 2 * n

    i = start
    while i < end:
        if not ext[i]:
            j = i
            while j < end and not ext[j]:
                j += 1

            gap_len = j - i
            prev_true = bool(ext[i - 1])
            next_true = bool(ext[j]) if j < ext.size else False

            if prev_true and next_true and gap_len <= max_gap:
                ext[i:j] = True
            i = j
        else:
            i += 1

    return ext[start:end]


def _longest_true_segment_circular(mask: np.ndarray) -> tuple[int, int]:
    """Return (start, length) of the longest contiguous True segment (circular)."""
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    if n == 0:
        return 0, 0

    mask2 = np.concatenate([mask, mask])
    best_len = 0
    best_start = 0
    cur_len = 0
    cur_start = 0

    for i, val in enumerate(mask2):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            # keep only segments that begin in the first copy and have length <= n
            if cur_len > best_len and cur_start < n and (i - cur_start) < n:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    return int(best_start), int(best_len)


def _stance_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tol_frac: float = 0.03,
    gap_frac: float = 0.03,
) -> tuple[np.ndarray, float, float, float]:
    """
    Returns:
      idx: stance indices (contiguous in crank parameter)
      duty: stance fraction
      dx: stride length during stance
      E: straightness error (max distance to best-fit line)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size

    y_min = float(y.min())
    y_max = float(y.max())
    eps = tol_frac * (y_max - y_min) + 1e-12

    mask = y <= y_min + eps
    mask = _close_gaps_circular(mask, max_gap=int(gap_frac * n))

    start, length = _longest_true_segment_circular(mask)
    idx = (np.arange(start, start + length) % n).astype(int)

    xs = x[idx]
    ys = y[idx]

    # best-fit line y = m x + q (least squares)
    X = np.vstack([xs, np.ones_like(xs)]).T
    m, q = np.linalg.lstsq(X, ys, rcond=None)[0]
    dist = np.abs(m * xs - ys + q) / np.sqrt(m * m + 1.0)

    dx = float(xs.max() - xs.min())
    duty = float(length / n)
    E = float(dist.max())
    return idx, duty, dx, E


def plot_footpath_overlay(
    pA: LinkageParams,
    pB: LinkageParams,
    outfile: str = "footpath_overlay.pdf",
    title: str = "Foot trajectory overlay (Design A vs Design B)",
    *,
    branch: int = +1,
    n_phi: int = 2000,
    tol_frac: float = 0.03,
    gap_frac: float = 0.03,
) -> None:
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    CA = point_C(phis, pA, branch=branch)
    CB = point_C(phis, pB, branch=branch)

    xA, yA = CA[:, 0], CA[:, 1]
    xB, yB = CB[:, 0], CB[:, 1]

    idxA, dutyA, dxA, EA = _stance_metrics(xA, yA, tol_frac=tol_frac, gap_frac=gap_frac)
    idxB, dutyB, dxB, EB = _stance_metrics(xB, yB, tol_frac=tol_frac, gap_frac=gap_frac)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    ax.plot(xA, yA, label=f"Design A  (Δx={dxA:.2f}l, E={EA:.3f}l, duty={dutyA:.2f})")
    ax.plot(xB, yB, label=f"Design B  (Δx={dxB:.2f}l, E={EB:.3f}l, duty={dutyB:.2f})")

    # highlight stance segments (thicker)
    ax.plot(xA[idxA], yA[idxA], linewidth=4.0, alpha=0.8)
    ax.plot(xB[idxB], yB[idxB], linewidth=4.0, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("x / l")
    ax.set_ylabel("y / l")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(loc="best", fontsize=9)

    # SAME axes for both (prevents autoscale hiding differences)
    xmin = min(xA.min(), xB.min())
    xmax = max(xA.max(), xB.max())
    ymin = min(yA.min(), yB.min())
    ymax = max(yA.max(), yB.max())
    pad = 0.05 * max(xmax - xmin, ymax - ymin)

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.plot(xA, yA, linestyle="--", alpha=0.7)
    ax.plot(xB, yB, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")  # vector PDF => crisp in LaTeX
    plt.close(fig)


# Example usage:
if __name__ == "__main__":
    pA = LinkageParams(a=1.13, b=1.32, c=1.20, d=1.46, l=1.0)
    pB = LinkageParams(a=1.97, b=1.92, c=1.52, d=2.26, l=1.0)
    plot_footpath_overlay(pA, pB, "footpath_overlay.pdf")
    print("Wrote footpath_overlay.pdf")