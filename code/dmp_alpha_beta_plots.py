# Plots showing the effect of the DMP parameters alpha_z and beta_z (see lerobot_dmp.py).
# Usage: python examples/dmp_alpha_beta_plots.py   -> PNG files in examples/dmp_plots/

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lerobot_dmp import DMP, DT, builtin_demo, get_robot, infinity_demo

OUT_DIR = os.path.join(os.path.dirname(__file__), "dmp_plots")

# Ordinal blue ramp for the alpha sweep (light -> dark = small -> large alpha), categorical slots for beta
ALPHA_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BETA_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DEMO_COLOR, GOAL_COLOR, INK, MUTED, GRID = "#898781", "#52514e", "#0b0b0b", "#898781", "#e1e0d9"

ALPHAS = [5.0, 10.0, 25.0, 50.0]
ALPHA_REF = 25.0
BETA_FACTORS = [(1 / 16, "suramorti, ζ = 2"), (1 / 4, "critique, ζ = 1"),
                (1.0, "sous-amorti, ζ = 0.5"), (4.0, "sous-amorti, ζ = 0.25")]

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "legend.frameon": False,
})


def sweeps():
    """(label, alpha_z, beta_z, color) for the alpha sweep then the beta sweep."""
    alpha_sweep = [(f"α = {a:g}, β = {a / 4:g}", a, a / 4, c) for a, c in zip(ALPHAS, ALPHA_COLORS)]
    beta_sweep = [(f"β = {ALPHA_REF * k:g} ({name})", ALPHA_REF, ALPHA_REF * k, c)
                  for (k, name), c in zip(BETA_FACTORS, BETA_COLORS)]
    return alpha_sweep, beta_sweep


def two_panels(title, subtitle):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.90, subtitle, ha="left", color=GOAL_COLOR)
    axes[0].set_title(f"α varie (β = α/4, amortissement critique)", loc="left", fontsize=10)
    axes[1].set_title(f"β varie (α = {ALPHA_REF:g})", loc="left", fontsize=10)
    return fig, axes


def save(fig, name):
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved '{path}'")


def spring_response():
    """Spring-damper at the core of the DMP, tau * dv = alpha (beta (g - y) - v): alpha sets speed, beta damping."""
    fig, axes = two_panels("1. Réponse du système ressort-amortisseur seul (τ·v̇ = α(β(g − y) − v), sans forçage)",
                           "Passage de y0 = 0 au but g = 1, τ = 1 s : α règle la rapidité, β/α règle les oscillations")
    t = np.arange(0, 2.0, DT)
    for ax, sweep in zip(axes, sweeps()):
        for label, a, b, color in sweep:
            y, v, ys = 0.0, 0.0, []
            for _ in t:
                ys.append(y)
                v += a * (b * (1.0 - y) - v) * DT
                y += v * DT
            ax.plot(t, ys, color=color, label=label)
        ax.axhline(1.0, color=GOAL_COLOR, linestyle="--", linewidth=1)
        ax.text(1.98, 1.02, "but g", ha="right", va="bottom", color=GOAL_COLOR)
        ax.set_xlabel("temps (s)")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("position y")
    save(fig, "1_reponse_ressort.png")


def new_goal(robot):
    """Pick demo replayed towards a new goal: the parameters shape how the movement adapts."""
    demo = builtin_demo(robot)
    new_g_z = 0.04
    fig, axes = two_panels("2. Généralisation à un nouveau but (mouvement « pick », hauteur z)",
                           f"Démo apprise vers z = {demo[-1, 3]:.2f} m, rejouée vers un nouveau but z = {new_g_z} m")
    t_demo = np.arange(len(demo)) * DT
    for ax, sweep in zip(axes, sweeps()):
        ax.plot(t_demo, demo[:, 3], color=DEMO_COLOR, linewidth=4, alpha=0.5, label="démonstration")
        for label, a, b, color in sweep:
            dmp = DMP(8, alpha_z=a, beta_z=b).fit(demo, DT)
            g = dmp.g.copy()
            g[3] = new_g_z
            traj = dmp.rollout(g=g)
            ax.plot(np.arange(len(traj)) * DT, traj[:, 3], color=color, label=label)
        ax.axhline(new_g_z, color=GOAL_COLOR, linestyle="--", linewidth=1)
        ax.text(0.05, new_g_z + 0.002, "nouveau but", va="bottom", color=GOAL_COLOR)
        ax.set_xlabel("temps (s)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("z (m)")
    save(fig, "2_nouveau_but.png")


def rollout_perturbed(dmp, t_push, push, dim, dt=DT, settle=0.5):
    """Same integration as DMP.rollout, with an instantaneous push of the state at t_push."""
    y, v, s = dmp.y0.copy(), np.zeros(dmp.n_dims), 1.0
    traj = [y.copy()]
    for k in range(int(round(dmp.tau * (1 + settle) / dt))):
        if k == int(t_push / dt):
            y[dim] += push
        psi = dmp._psi(s)
        f = dmp.w @ psi / (psi.sum() + 1e-10) * s
        dv = (dmp.K * (dmp.g - y) - dmp.D * v - dmp.K * (dmp.g - dmp.y0) * s + dmp.K * f) / dmp.tau
        v = v + dv * dt
        y = y + v / dmp.tau * dt
        s = s - dmp.alpha_s * s / dmp.tau * dt
        traj.append(y.copy())
    return np.array(traj)


def perturbation(robot):
    """Infinity sign with a push on the base rotation at t = 3 s: alpha/beta set how it recovers."""
    demo = infinity_demo(robot)
    t_push, push = 3.0, 0.3
    fig, axes = two_panels("3. Rejet d'une perturbation (signe ∞, rotation de la base)",
                           f"Poussée de +{push} rad sur la base à t = {t_push:g} s : retour vers la trajectoire apprise")
    t_demo = np.arange(len(demo)) * DT
    for ax, sweep in zip(axes, sweeps()):
        ax.plot(t_demo, demo[:, 0], color=DEMO_COLOR, linewidth=4, alpha=0.5, label="démonstration")
        for label, a, b, color in sweep:
            dmp = DMP(8, alpha_z=a, beta_z=b).fit(demo, DT)
            traj = rollout_perturbed(dmp, t_push, push, dim=0)
            ax.plot(np.arange(len(traj)) * DT, traj[:, 0], color=color, label=label)
        ax.axvline(t_push, color=GOAL_COLOR, linestyle=":", linewidth=1)
        ax.text(t_push + 0.05, ax.get_ylim()[1] * 0.95, "poussée", va="top", color=GOAL_COLOR)
        ax.set_xlabel("temps (s)")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("rotation de la base (rad)")
    save(fig, "3_perturbation.png")


def reproduction(robot):
    """Reproducing the demo itself: the learned weights compensate, curves are almost identical."""
    demo = infinity_demo(robot)
    fig, axes = two_panels("4. Reproduction de la démonstration (signe ∞ vu de face)",
                           "Sans changement de but ni perturbation, les poids appris compensent α et β : les courbes se superposent")
    for ax, sweep in zip(axes, sweeps()):
        ax.plot(demo[:, 0], demo[:, 3], color=DEMO_COLOR, linewidth=6, alpha=0.4, label="démonstration")
        for label, a, b, color in sweep:
            traj = DMP(8, alpha_z=a, beta_z=b).fit(demo, DT).rollout()
            err = np.abs(traj[:len(demo), 3] - demo[:, 3]).max() * 1000
            ax.plot(traj[:, 0], traj[:, 3], color=color, linewidth=1.5, label=f"{label} — écart max {err:.1f} mm")
        ax.set_xlabel("rotation de la base (rad)")
        ax.legend(loc="upper center", fontsize=7.5, bbox_to_anchor=(0.5, -0.18), ncol=1)
    axes[0].set_ylabel("z (m)")
    fig.set_size_inches(12, 6)
    save(fig, "4_reproduction.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    robot = get_robot("so100")
    spring_response()
    new_goal(robot)
    perturbation(robot)
    reproduction(robot)


if __name__ == "__main__":
    main()
