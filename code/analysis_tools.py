from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from linkage import LinkageParams, point_C, transmission_angle_mu, is_fully_feasible


def close_gaps_circular(mask: np.ndarray, max_gap: int) -> np.ndarray:
#little cover for false runs
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    if n == 0:
        return mask

    ext = np.concatenate([mask, mask, mask])
    start = n
    end = 2 * n

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


def longest_true_segment_circular(mask: np.ndarray) -> Tuple[int, int]:
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

            # Keep only segments that begin in the first copy and have length <= n
            if cur_len > best_len and cur_start < n and (i - cur_start) < n:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    return int(best_start), int(best_len)


@dataclass(frozen=True)
class StanceMetrics:
    duty: float
    dx: float
    straightness_E: float
    y_min: float
    y_max: float


def stance_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tol_frac: float = 0.03,
    gap_frac: float = 0.03,
) -> Tuple[np.ndarray, StanceMetrics]:

    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size

    y_min = float(y.min())
    y_max = float(y.max())
    eps = tol_frac * (y_max - y_min) + 1e-12

    mask = y <= y_min + eps
    mask = close_gaps_circular(mask, max_gap=int(gap_frac * n))

    start, length = longest_true_segment_circular(mask)
    idx = (np.arange(start, start + length) % n).astype(int)

    # Straightness error: max distance to best-fit line
    xs = x[idx]
    ys = y[idx]
    X = np.vstack([xs, np.ones_like(xs)]).T
    m, q = np.linalg.lstsq(X, ys, rcond=None)[0]
    dist = np.abs(m * xs - ys + q) / np.sqrt(m * m + 1.0)

    dx = float(xs.max() - xs.min())
    duty = float(length / n)
    E = float(dist.max())

    return idx, StanceMetrics(duty=duty, dx=dx, straightness_E=E, y_min=y_min, y_max=y_max)


@dataclass(frozen=True)
class DesignScore:
    params: LinkageParams
    score: float
    duty: float
    dx: float
    E: float
    swing_height: float
    min_mu: float


def evaluate_design(
    p: LinkageParams,
    *,
    branch: int = +1,
    n_phi: int = 1500,
    tol_frac: float = 0.03,
    gap_frac: float = 0.03,
    min_duty: float = 0.30,
    min_mu_deg: float = 0.0,
) -> Optional[DesignScore]:

    if not is_fully_feasible(p, branch=branch, n_phi=n_phi):
        return None

    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    C = point_C(phis, p, branch=branch)

    x = C[:, 0]
    y = C[:, 1]

    idx, m = stance_metrics(x, y, tol_frac=tol_frac, gap_frac=gap_frac)
    if m.duty < min_duty:
        return None

    swing_height = float(m.y_max - (m.y_min + tol_frac * (m.y_max - m.y_min)))

    mu = transmission_angle_mu(phis, p, branch=branch)
    min_mu = float(mu.min())
    if min_mu < min_mu_deg:
        return None

    # A small, transparent scoring function.
    score = (
        1.0 * m.dx
        - 2.5 * m.straightness_E
        + 0.3 * swing_height
        + 0.4 * m.duty
        + 0.01 * min_mu
    )

    return DesignScore(
        params=p, 
        score=float(score),
        duty=m.duty,
        dx=m.dx,
        E=m.straightness_E,
        swing_height=swing_height,
        min_mu=min_mu,
    )
