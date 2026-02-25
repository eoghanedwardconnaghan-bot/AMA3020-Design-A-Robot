"""
Draw the planar linkage diagram (O, O', A, B, C) like the figure you shared.

Requires: numpy, matplotlib
Run: python draw_linkage.py
Output: linkage.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc


def circle_circle_intersection(c0, r0, c1, r1):
    """
    Return the two intersection points of circles (c0,r0) and (c1,r1),
    or [] if they don't intersect.
    """
    c0 = np.asarray(c0, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    dist = np.linalg.norm(c1 - c0)
    if dist == 0 or dist > r0 + r1 or dist < abs(r0 - r1):
        return []

    a = (r0**2 - r1**2 + dist**2) / (2 * dist)
    h2 = r0**2 - a**2
    h = np.sqrt(max(h2, 0.0))

    p = c0 + a * (c1 - c0) / dist
    perp = np.array([-(c1 - c0)[1], (c1 - c0)[0]]) / dist

    p1 = p + h * perp
    p2 = p - h * perp
    return [p1, p2]


def draw_linkage(
    l=2.6,      # |OO'|
    a=1.2,      # crank OA
    b=3.4,      # crank O'B
    c=2.0,      # coupler AB
    d=1.0,      # extension BC (measured from B)
    phi_deg=35, # input angle at O
    elbow="up", # "up" or "down" branch for B
    outfile="linkage.pdf",
):
    # --- points ---
    O = np.array([0.0, 0.0])
    Op = np.array([-l, 0.0])  # O'
    phi = np.deg2rad(phi_deg)

    A = O + a * np.array([np.cos(phi), np.sin(phi)])

    # B from circle-circle intersection
    ints = circle_circle_intersection(Op, b, A, c)
    if not ints:
        raise ValueError("No real assembly: circles do not intersect for these parameters.")

    B = max(ints, key=lambda p: p[1]) if elbow == "up" else min(ints, key=lambda p: p[1])

    # C on extension of BA beyond B:  C = B + (d/c) (B - A)
    C = B + (d / c) * (B - A)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    # baseline (dashed)
    ax.plot([Op[0] - 0.2, O[0] + 1.5], [0, 0],
            linestyle=(0, (4, 4)), linewidth=2, color="black")

    # dotted circle about O (radius a)
    ax.add_patch(Circle(O, a, fill=False, linestyle=(0, (1, 3)),
                        linewidth=2, color="gray"))

    # links (thick)
    ax.plot([O[0], A[0]],  [O[1], A[1]],  linewidth=3, color="black")  # a
    ax.plot([Op[0], B[0]], [Op[1], B[1]], linewidth=3, color="black")  # b
    ax.plot([A[0], B[0]],  [A[1], B[1]],  linewidth=3, color="black")  # c
    ax.plot([B[0], C[0]],  [B[1], C[1]],  linewidth=3, color="black")  # d

    # joints: open circles at O, O', A, B
    def open_joint(P, r=0.06):
        ax.add_patch(Circle(P, r, facecolor="white", edgecolor="black",
                            linewidth=2, zorder=5))

    for P in [Op, O, A, B]:
        open_joint(P)

    # C: filled dot
    ax.add_patch(Circle(C, 0.06, facecolor="black", edgecolor="black",
                        linewidth=1.5, zorder=6))

    # angle arc for phi at O
    arc_r = 0.55
    ax.add_patch(Arc(O, 2 * arc_r, 2 * arc_r, angle=0, theta1=0, theta2=phi_deg,
                     linewidth=2, color="black"))
    ax.text(O[0] + 0.65, O[1] + 0.20, r"$\phi$", fontsize=18)

    # labels for points
    ax.text(C[0] - 0.15, C[1] + 0.10, r"$C$", fontsize=18)
    ax.text(B[0] + 0.10, B[1] + 0.05, r"$B$", fontsize=18)
    ax.text(A[0] + 0.10, A[1] - 0.05, r"$A$", fontsize=18)
    ax.text(O[0], O[1] - 0.28, r"$O$", fontsize=18, ha="center")
    ax.text(Op[0], Op[1] - 0.28, r"$O'$", fontsize=18, ha="center")

    # helper for segment text near midpoints (with "prefer_up" option)
    def label_segment(P, Q, text, offset=0.12, prefer_up=False):
        P = np.asarray(P); Q = np.asarray(Q)
        mid = 0.5 * (P + Q)
        v = Q - P

        # perpendicular offset
        n = np.array([-v[1], v[0]])
        n = n / (np.linalg.norm(n) + 1e-12)

        # force the label to go "above" (positive y) if requested
        if prefer_up and n[1] < 0:
            n = -n

        ax.text(mid[0] + offset * n[0], mid[1] + offset * n[1], text, fontsize=18)

    # length labels
    label_segment(O, A,  r"$a$", offset=0.14)
    label_segment(Op, B, r"$b$", offset=0.14)
    label_segment(A, B,  r"$c$", offset=0.16, prefer_up=True)  # moved off/above the link
    label_segment(B, C,  r"$d$", offset=0.16, prefer_up=True)  # moved off/above the link

    # label l on baseline between O' and O
    ax.text(0.5 * (Op[0] + O[0]), 0.12, r"$l$", fontsize=18, ha="center")

    # styling
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # view window
    xs = [Op[0], O[0], A[0], B[0], C[0]]
    ys = [Op[1], O[1], A[1], B[1], C[1]]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)

    # save or show
    plt.savefig(outfile, dpi=1300, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    draw_linkage()