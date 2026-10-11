"""Figuras estáticas del README (docs/img/*.png), a partir de results/*.json.

    python make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import varnet as V

INK, MUTED, GRID = "#131a22", "#73808d", "#e6ebf0"
BLUE, ORANGE, GREEN, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "legend.frameon": False, "savefig.dpi": 160,
                     "savefig.bbox": "tight"})


def load(name):
    return json.load(open(f"results/{name}.json"))


def convergence():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, label, color in (("proj_orig", "Original", ORANGE), ("proj_traj_raw", "Red nueva + pérdida cruda", YELLOW),
                               ("proj_orignet_pos", "Red original + pérdida de Picard", GREEN),
                               ("proj_traj_pos", "Red nueva + pérdida de Picard", BLUE)):
        hist = load(name)["hist"]
        ax.loglog(hist["step"], hist["e_vi"], color=color, lw=2, label=label)
    ax.set(xlabel="pasos de entrenamiento", ylabel="RMSE (m)", title="Tiro parabólico: error contra pasos")
    ax.legend()
    fig.savefig("docs/img/convergencia.png"); plt.close(fig)


def ablation():
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sizes = (25, 50, 100, 200)
    for name, label, color in (("raw", "Cruda  ‖r‖²", ORANGE), ("mom", "Momento  ‖π‖²", YELLOW),
                               ("posmom", "Picard + momento", GREEN), ("pos", "Picard  ‖c‖²", BLUE)):
        ax.loglog(sizes, [load(f"pendN{N}_{name}")["res"]["rmse_vi"] for N in sizes], "o-", color=color, lw=2, label=label)
    ax.set_xticks(sizes, [str(N) for N in sizes]); ax.minorticks_off()
    ax.set(xlabel="N (intervalos de tiempo)", ylabel="RMSE final (rad)", title="Péndulo: mismo presupuesto, distinta pérdida")
    ax.legend(loc="center right")
    fig.savefig("docs/img/ablacion_N.png"); plt.close(fig)


def zero_shot():
    potentials = (("henon", "Hénon–Heiles", (-1.0, 1.0, -0.8, 1.2)), ("eggcrate", "Red de cosenos", (-6, 6, -6, 6)),
                  ("gauss3", "Tres pozos gaussianos", (-3.5, 3.5, -3.5, 3.5)),
                  ("kepler", "Kepler suavizado", (-3.5, 3.5, -3.5, 3.5)), ("duffing", "Doble pozo", (-1.8, 1.8, -1.8, 1.8)))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    for ax, (key, title, box) in zip(axes, potentials):
        res = load(f"unixl_{key}")["res"]
        x, y = np.linspace(box[0], box[1], 120), np.linspace(box[2], box[3], 120)
        Z = V.POTENTIALS[key]().V(torch.tensor(np.stack(np.meshgrid(x, y), -1)), torch.zeros(1, 0)).numpy()
        ax.contour(x, y, Z, levels=np.linspace(np.quantile(Z, 0.02), np.quantile(Z, 0.9), 10), colors=MUTED,
                   linewidths=0.5, alpha=0.6)
        n = int(8.0 / (2 * 1.0 / len(res["err_vi_t"])))        # hasta t = 8
        for i, color in enumerate((BLUE, ORANGE, GREEN, VIOLET)):
            nn, rk = np.array(res["ex_nn"][i])[:n], np.array(res["ex_rk"][i])[:n]
            ax.plot(rk[:, 0], rk[:, 1], color=INK, lw=0.9, ls=(0, (2, 2)))
            ax.plot(nn[:, 0], nn[:, 1], color=color, lw=1.6)
            ax.plot(*res["ex_q0"][i], "o", color=color, ms=4)
        ax.set(xlim=box[:2], ylim=box[2:], title=title, xticks=[], yticks=[]); ax.set_aspect("equal"); ax.grid(False)
    fig.suptitle("Una sola red, cinco potenciales que nunca vio (color: red; punteado: RK4; hasta t = 8)", y=1.02)
    fig.savefig("docs/img/potenciales_nunca_vistos.png"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs("docs/img", exist_ok=True)
    convergence(); ablation(); zero_shot()
    print("docs/img:", ", ".join(sorted(os.listdir("docs/img"))))
