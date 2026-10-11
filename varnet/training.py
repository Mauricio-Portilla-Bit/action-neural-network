"""Entrenamiento sin datos: cada paso muestrea condiciones iniciales nuevas."""
import math
import time

import torch

from .losses import loss_fn
from .mechanics import path_quantities, picard, velocity_from_momentum
from .models import TrajNet
from .reference import rk4_reference, vi_reference


def make_testset(system, n, N, T, K=1, device="cpu", seed=1234):
    """Conjunto de prueba fijo con sus referencias en float64.

    q_vi: integrador variacional sobre una ventana. q_rk: RK4 sobre K ventanas.
    """
    gen = torch.Generator().manual_seed(seed)
    q0, v0, th = (x.to(device) for x in system.sample(n, torch.device("cpu"), torch.float64, gen))
    dt = T / N
    q_vi, _ = vi_reference(system, q0, v0, th, dt, N)
    q_rk, _ = rk4_reference(system, q0, v0, th, dt, N * K)
    return dict(q0=q0.cpu(), v0=v0.cpu(), th=th.cpu(), q_vi=q_vi.cpu(), q_rk=q_rk.cpu(), K=K)


@torch.no_grad()
def evaluate(model, testset, device, dtype):
    """RMSE de una ventana contra el integrador variacional y contra RK4."""
    q0, v0, th = (testset[k].to(device, dtype) for k in ("q0", "v0", "th"))
    Q = model(q0, v0, th).double().cpu()
    rmse = lambda ref: (Q - ref[:, :Q.shape[1]]).pow(2).sum(-1).mean().sqrt().item()
    return rmse(testset["q_vi"]), rmse(testset["q_rk"])


def train(system, model, loss="pos", steps=20000, batch=4096, lr=2e-3, warmup=300, clip=None, beta=1.0,
          orig_opt=False, device="cpu", dtype=torch.float32, testset=None, eval_every=500, seed=0,
          log=print, ckpt=None):
    """Entrena `model` y devuelve el historial.

    loss: una de `losses.LOSSES`, o "sgK" para Picard con semigradiente: regresión hacia el blanco
    fijo Φᴷ(q), sin derivar a través de la física.
    orig_opt: usa el optimizador del proyecto original (AdamW, lr 3e-3, coseno, recorte 1.0).
    """
    torch.manual_seed(seed)
    model = model.to(device, dtype)
    if orig_opt:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
        clip = 1.0
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        cosine = lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * cosine(s))
    q_scale = system.scales()[0]
    has_velocity_head = isinstance(model, TrajNet)
    hist = dict(step=[], time=[], loss=[], e_vi=[], e_rk=[])
    t0 = time.time()
    for step in range(1, steps + 1):
        q0, v0, th = system.sample(batch, device, dtype)
        Q, vN = model(q0, v0, th, return_v=True) if has_velocity_head else (model(q0, v0, th), None)
        if loss.startswith("sg"):
            target = picard(system, q0, v0, th, Q, model.dt, int(loss[2:] or 1))
            value = (((Q - target) / q_scale) ** 2).mean()
            pq = path_quantities(system, q0, v0, th, Q.detach(), model.dt, create_graph=False)
        else:
            pq = path_quantities(system, q0, v0, th, Q, model.dt)
            value = loss_fn(system, pq, model.dt, loss, q0, v0, th, beta=beta)
        if has_velocity_head:
            # La velocidad final se ajusta al balance de impulso del propio camino. El blanco no
            # recibe gradiente, así que este término no altera la física de los nodos.
            v_target = velocity_from_momentum(system, Q[:, -1].detach(), pq["pN_int"].detach(), th)
            value = value + (((vN - v_target) * model.T / q_scale) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        value.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); sched.step()
        if step % eval_every == 0 or step == 1 or step == steps:
            elapsed = time.time() - t0
            e_vi, e_rk = evaluate(model, testset, device, dtype) if testset else (float("nan"),) * 2
            for key, x in zip(hist, (step, elapsed, value.item(), e_vi, e_rk)):
                hist[key].append(x)
            if ckpt and len(hist["step"]) % 5 == 0:
                torch.save(model.state_dict(), ckpt)
            log(f"[{system.name}/{loss}] paso {step:>7d} | pérdida {value.item():.3e} | "
                f"RMSE vs VI {e_vi:.3e} | vs RK4 {e_rk:.3e} | {elapsed:.0f}s")
    return hist
