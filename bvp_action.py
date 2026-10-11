"""¿Se puede usar la acción directamente como pérdida?

Sólo en un problema de contorno (q0 y qT fijos) y sólo antes del primer punto conjugado. Para el
oscilador armónico L = ½q'² − ½ω²q² con T = 1, la acción es un mínimo si ωT < π y un punto de
silla si ωT > π: en el segundo caso «minimizarla» diverge.

    python bvp_action.py      ->  results/bvp_action.json
"""
import json
import math
import os

import torch
import torch.nn as nn

N, T = 50, 1.0
dt = T / N


def action(q0, qT, inner, w2):
    path = torch.cat([q0, inner, qT], 1)
    mid, vel = 0.5 * (path[:, 1:] + path[:, :-1]), (path[:, 1:] - path[:, :-1]) / dt
    return (dt * (0.5 * vel ** 2 - 0.5 * w2 * mid ** 2)).sum(1)


def exact(q0, qT, w):
    t = dt * torch.arange(1, N)[None]
    return (q0 * torch.sin(w * (T - t)) + qT * torch.sin(w * t)) / torch.sin(w * T)


def run(w_min, w_max, steps=4000, batch=1024, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, N - 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    s = torch.linspace(0, 1, N + 1)[None, 1:-1]
    hist = dict(step=[], S=[], rmse=[])
    for step in range(1, steps + 1):
        q0, qT = torch.rand(batch, 1) * 2 - 1, torch.rand(batch, 1) * 2 - 1
        w = w_min + (w_max - w_min) * torch.rand(batch, 1)
        # Los extremos se imponen por construcción: recta + s(1 − s)·red.
        inner = q0 + (qT - q0) * s + s * (1 - s) * net(torch.cat([q0, qT, w / 5], 1))
        S = action(q0, qT, inner, w ** 2).mean()
        opt.zero_grad(); S.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            hist["step"].append(step)
            hist["S"].append(S.item())
            hist["rmse"].append((inner - exact(q0, qT, w)).pow(2).mean().sqrt().item())
            if not math.isfinite(S.item()) or abs(S.item()) > 1e6:
                break
    return hist


if __name__ == "__main__":
    results = {"antes del punto conjugado (ωT ∈ [0.5, 2.5])": run(0.5, 2.5),
               "después (ωT ∈ [3.6, 4.6])": run(3.6, 4.6)}
    for label, hist in results.items():
        print(f"{label} | S final {hist['S'][-1]:.3g} | RMSE final {hist['rmse'][-1]:.3g} | pasos {hist['step'][-1]}")
    os.makedirs("results", exist_ok=True)
    with open("results/bvp_action.json", "w") as f:
        json.dump(results, f)
