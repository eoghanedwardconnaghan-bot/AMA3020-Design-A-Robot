"""Four-bar linkage with a coupler extension (planar).

This module implements the forward kinematics for the mechanism:

    O --(a)--> A --(c)--> B --(b)--> O'
                           |
                          (d)
                           |
                           C

with O=(0,0), O'=(l,0) in the body frame.

The key closed-form step is computing point B as a circle-circle intersection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


@dataclass(frozen=True)
class LinkageParams:
    """Link lengths for the four-bar + extension."""

    a: float  # |OA|
    b: float  # |O'B|
    c: float  # |AB|
    d: float  # |BC|
    l: float = 1.0  # |OO'| (scale)


AssemblyMode = Literal[+1, -1]


def point_A(phi: np.ndarray, p: LinkageParams) -> np.ndarray:
    """Compute A(phi) for array-like phi (radians)."""
    phi = np.asarray(phi)
    return np.column_stack([p.a * np.cos(phi), p.a * np.sin(phi)])


def circle_circle_B(
    phi: np.ndarray,
    p: LinkageParams,
    branch: AssemblyMode = +1,
    *,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    phi = np.asarray(phi)

    Op = np.array([p.l, 0.0])
    A = point_A(phi, p)

    Delta = A - Op
    r = np.linalg.norm(Delta, axis=1)

    
    r = np.maximum(r, eps)

    ex = Delta / r[:, None]
    ey = np.column_stack([-ex[:, 1], ex[:, 0]])

    alpha = (p.b * p.b - p.c * p.c + r * r) / (2.0 * r)
    radicand = p.b * p.b - alpha * alpha

    beta = branch * np.sqrt(np.maximum(radicand, 0.0))
    B = Op + alpha[:, None] * ex + beta[:, None] * ey

    return A, B, radicand


def point_C(phi: np.ndarray, p: LinkageParams, branch: AssemblyMode = +1) -> np.ndarray:
    A, B, rad = circle_circle_B(phi, p, branch)
    # C = B + (d/c)(B - A)
    return B + (p.d / p.c) * (B - A)


def transmission_angle_mu(phi: np.ndarray, p: LinkageParams, branch: AssemblyMode = +1) -> np.ndarray:

    phi = np.asarray(phi)
    Op = np.array([p.l, 0.0])
    A, B, _ = circle_circle_B(phi, p, branch)

    BA = A - B
    BOp = Op - B

    num = np.sum(BA * BOp, axis=1)
    den = np.linalg.norm(BA, axis=1) * np.linalg.norm(BOp, axis=1)
    den = np.maximum(den, 1e-12)

    cos_mu = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cos_mu))


def is_fully_feasible(
    p: LinkageParams,
    branch: AssemblyMode = +1,
    *,
    n_phi: int = 2000,
    tol: float = 1e-8,
) -> bool:
    """Check feasibility (real assembly) over a full input rotation."""
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    _, _, rad = circle_circle_B(phis, p, branch)
    return bool(np.all(rad >= -tol))
