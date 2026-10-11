"""Soluciones de referencia en doble precisión, genéricas en el Lagrangiano."""
import torch
from torch.func import grad, hessian, jacrev, vmap

from .mechanics import apply_Minv, dL, momentum, velocity_from_momentum


@torch.no_grad()
def vi_reference(system, q0, v0, th, dt, N, iters=60):
    """Integrador variacional de punto medio: la solución exacta del problema discreto.

    Es el mínimo común de todas las pérdidas. Cada paso implícito se resuelve por punto fijo
    precondicionado con M⁻¹. Devuelve los nodos (B, N, d) y el momento final.
    """
    q, v, p = q0.clone(), v0.clone(), momentum(system, q0, v0, th)
    nodes = []
    for _ in range(N):
        q_next = q + dt * v
        for _ in range(iters):   # resuelve p_k + D1 L_d(q_k, q_{k+1}) = 0
            mid, w = 0.5 * (q + q_next), (q_next - q) / dt
            dq, dv, _ = dL(system, mid, w, th)
            step = dt * apply_Minv(system, mid, w, th, (p + 0.5 * dt * dq - dv).detach())
            q_next = (q_next + step).detach()
            if step.abs().max() < 1e-14:
                break
        mid, w = 0.5 * (q + q_next), (q_next - q) / dt
        dq, dv, _ = dL(system, mid, w, th)
        p = (0.5 * dt * dq + dv).detach()   # p_{k+1} = D2 L_d(q_k, q_{k+1})
        q = q_next
        v = velocity_from_momentum(system, q, p, th)
        nodes.append(q)
    return torch.stack(nodes, 1), p


def acceleration_fn(system):
    """Aceleración de Euler–Lagrange, a = M⁻¹(∂L/∂q − ∂²L/∂v∂q · v)."""
    if system.separable:
        return lambda q, v, th: dL(system, q, v, th)[0].detach()

    def single(q, v, th):
        M = hessian(system.L, argnums=1)(q, v, th)
        C = jacrev(grad(system.L, argnums=1), argnums=0)(q, v, th)
        return torch.linalg.solve(M, grad(system.L, argnums=0)(q, v, th) - C @ v)

    return vmap(single)


@torch.no_grad()
def rk4_reference(system, q0, v0, th, dt, N, sub=4):
    """Solución continua: RK4 con paso dt/sub. Devuelve posiciones y velocidades (B, N, d)."""
    acc = acceleration_fn(system)
    h = dt / sub
    q, v = q0.clone(), v0.clone()
    qs, vs = [], []
    for _ in range(N):
        for _ in range(sub):
            k1q, k1v = v, acc(q, v, th)
            k2q, k2v = v + 0.5 * h * k1v, acc(q + 0.5 * h * k1q, v + 0.5 * h * k1v, th)
            k3q, k3v = v + 0.5 * h * k2v, acc(q + 0.5 * h * k2q, v + 0.5 * h * k2v, th)
            k4q, k4v = v + h * k3v, acc(q + h * k3q, v + h * k3v, th)
            q = q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        qs.append(q); vs.append(v)
    return torch.stack(qs, 1), torch.stack(vs, 1)
