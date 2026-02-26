import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def circle_circle_intersections(P0, r0, P1, r1):
    P0 = np.asarray(P0, float)
    P1 = np.asarray(P1, float)

    d = np.linalg.norm(P1 - P0)
    if d == 0 or d > r0 + r1 or d < abs(r0 - r1):
        return None

    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h2 = r0**2 - a**2
    if h2 < 0:
        return None
    h = np.sqrt(max(h2, 0.0))

    P2 = P0 + a * (P1 - P0) / d
    v = (P1 - P0) / d
    perp = np.array([-v[1], v[0]])

    X1 = P2 + h * perp
    X2 = P2 - h * perp
    return X1, X2

def choose_branch(prev_B, B1, B2):
    
    if prev_B is None:
        return B1 if B1[1] >= B2[1] else B2
    return B1 if np.linalg.norm(B1 - prev_B) <= np.linalg.norm(B2 - prev_B) else B2


def simulate_linkage(l, a, b, c, d, n_steps=1200):
    O_star = np.array([0.0, 0.0])
    O      = np.array([l, 0.0])

    phis = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
    A = np.zeros((n_steps, 2))
    B = np.full((n_steps, 2), np.nan)
    C = np.full((n_steps, 2), np.nan)

    prev_B = None

    for i, phi in enumerate(phis):
        Ai = O + a * np.array([np.cos(phi), np.sin(phi)])
        A[i] = Ai

        inter = circle_circle_intersections(O_star, b, Ai, c)
        if inter is None:
            prev_B = None
            continue

        B1, B2 = inter
        Bi = choose_branch(prev_B, B1, B2)
        B[i] = Bi
        prev_B = Bi

        
        C[i] = Bi + (d / c) * (Bi - Ai)

    return O_star, O, phis, A, B, C


def pick_params(l=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    
    c = rng.uniform(0.40*l, 0.95*l)
    d = rng.uniform(0.40*l, 1.10*l)

    for _ in range(5000):
        a = rng.uniform(0.15*l, 0.65*l)
        b = rng.uniform(0.35*l, 1.30*l)

        
        d_min = abs(l - a)
        d_max = l + a
        if abs(b - c) <= d_min and (b + c) >= d_max:
            return a, b, c, d

    
    return 0.35*l, 0.55*l, 0.45*l, 0.60*l


rng = np.random.default_rng()

l = 1.0
a, b, c, d = pick_params(l, rng=rng)

print(f"Using l={l:.3f}, a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")

O_star, O, phis, A, B, C = simulate_linkage(l, a, b, c, d, n_steps=1400)


good = ~np.isnan(B[:, 0])
A_g = A[good]
B_g = B[good]
C_g = C[good]

err_OA = np.max(np.abs(np.linalg.norm(A_g - O, axis=1) - a))
err_OB = np.max(np.abs(np.linalg.norm(B_g - O_star, axis=1) - b))
err_AB = np.max(np.abs(np.linalg.norm(B_g - A_g, axis=1) - c))
err_BC = np.max(np.abs(np.linalg.norm(C_g - B_g, axis=1) - d))

print("Max constraint errors:")
print(" |OA|-a :", err_OA)
print(" |O'B|-b:", err_OB)
print(" |AB|-c :", err_AB)
print(" |BC|-d :", err_BC)


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.set_title("Linkage animation (OA, O'B, AB, BC) with constraints satisfied")
ax.set_xlabel("x")
ax.set_ylabel("y")


ax.plot(C[:, 0], C[:, 1], linewidth=1, alpha=0.25, label="C path (trace)")


ax.scatter([O_star[0], O[0]], [O_star[1], O[1]], s=90)
ax.text(O_star[0], O_star[1], "O'", ha="right", va="bottom")
ax.text(O[0], O[1], "O", ha="left", va="bottom")


ptA = ax.scatter([], [], s=60)
ptB = ax.scatter([], [], s=60)
ptC = ax.scatter([], [], s=60)


line_OA, = ax.plot([], [], linewidth=3, label="OA")
line_OB, = ax.plot([], [], linewidth=3, label="O'B")
line_AB, = ax.plot([], [], linewidth=3, label="AB")
line_BC, = ax.plot([], [], linewidth=3, label="BC")


all_xy = np.vstack([A[good], B[good], C[good], O_star[None, :], O[None, :]])
pad = 0.2
ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
ax.legend(loc="best")

def init():
    return ptA, ptB, ptC, line_OA, line_OB, line_AB, line_BC

def update(frame):
    Ai = A[frame]
    Bi = B[frame]
    Ci = C[frame]

    
    ptA.set_offsets([Ai[0], Ai[1]])
    line_OA.set_data([O[0], Ai[0]], [O[1], Ai[1]])


    if np.isnan(Bi[0]) or np.isnan(Ci[0]):
        ptB.set_offsets([np.nan, np.nan])
        ptC.set_offsets([np.nan, np.nan])
        line_OB.set_data([], [])
        line_AB.set_data([], [])
        line_BC.set_data([], [])
    else:
        ptB.set_offsets([Bi[0], Bi[1]])
        ptC.set_offsets([Ci[0], Ci[1]])

        line_OB.set_data([O_star[0], Bi[0]], [O_star[1], Bi[1]])
        line_AB.set_data([Ai[0], Bi[0]], [Ai[1], Bi[1]])
        line_BC.set_data([Bi[0], Ci[0]], [Bi[1], Ci[1]])

    return ptA, ptB, ptC, line_OA, line_OB, line_AB, line_BC

ani = FuncAnimation(fig, update, frames=len(phis), init_func=init, interval=20, blit=True)


try:
    ani.save("linkage_animation.gif", writer="pillow", fps=30)
    print("Saved: linkage_animation.gif")
except Exception as e:
    print("Could not save GIF (install pillow with: python -m pip install pillow)")
    print("Error:", e)


plt.show()
