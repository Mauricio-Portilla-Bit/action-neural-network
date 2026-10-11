"""Mecánica variacional discreta, genérica en el Lagrangiano.

Con el Lagrangiano discreto de punto medio

    L_d(q_k, q_{k+1}) = Δt · L( (q_k + q_{k+1})/2 , (q_{k+1} − q_k)/Δt )

las ecuaciones de Euler–Lagrange discretas con dato inicial (q0, p0) son

    r_0 = p0 + D1 L_d(q0, q1)                           = 0     (Legendre discreta)
    r_k = D2 L_d(q_{k-1}, q_k) + D1 L_d(q_k, q_{k+1})   = 0     (k = 1..N-1)

r es el gradiente de la acción respecto al camino. Su suma acumulada telescopa al balance
impulso–momento π, e integrada otra vez da el defecto de Picard c = Φ(q) − q:

    π_k = Σ_{j≤k} r_j = p0 + Δt Σ_{j<k} ∂L/∂q|_j + (Δt/2) ∂L/∂q|_k − ∂L/∂v|_k
    c   = Δt · cumsum(M⁻¹ π)

r, π y c se anulan en la misma trayectoria; lo que cambia es el condicionamiento de la pérdida
que se construye con cada uno (~N⁴, ~N² y ~1).
"""
import torch
from torch.func import hessian, vmap


def dL(system, q, v, th, create_graph=False):
    """-> (∂L/∂q, ∂L/∂v, L) evaluados en (q, v). Funciona también dentro de `torch.no_grad()`."""
    with torch.enable_grad():
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        if not v.requires_grad:
            v = v.detach().requires_grad_(True)
        L = system.L(q, v, th)
        dq, dv = torch.autograd.grad(L.sum(), [q, v], create_graph=create_graph, allow_unused=True)
    if dq is None:   # L no depende de q
        dq = torch.zeros_like(q)
    return dq, dv, L


def mass_matrix(system, q, v, th):
    """M = ∂²L/∂v², de forma (..., d, d)."""
    batch = q.shape[:-1]
    qf, vf = q.reshape(-1, system.d), v.reshape(-1, system.d)
    thf = th.expand(*batch, th.shape[-1]).reshape(-1, th.shape[-1])
    M = vmap(hessian(system.L, argnums=1))(qf, vf, thf)
    return M.reshape(*batch, system.d, system.d)


def apply_Minv(system, q, v, th, x):
    """M(q)⁻¹ x. M se trata como constante: sólo actúa como precondicionador."""
    if system.unit_mass:
        return x
    if hasattr(system, "Minv"):
        return system.Minv(q.detach(), th, x)
    M = mass_matrix(system, q.detach(), v.detach(), th)
    return torch.linalg.solve(M, x.unsqueeze(-1)).squeeze(-1)


def momentum(system, q, v, th):
    """p = ∂L/∂v."""
    return dL(system, q, v, th)[1].detach()


def velocity_from_momentum(system, q, p, th):
    """Inversa de la transformada de Legendre para L cuadrático en v: v = M⁻¹(p − ∂L/∂v|_{v=0})."""
    zero = torch.zeros_like(p)
    offset = dL(system, q, zero, th)[1].detach()
    return apply_Minv(system, q, zero, th, p - offset)


def energy(system, q, v, th):
    """E = v·∂L/∂v − L."""
    _, dv, L = dL(system, q, v, th)
    return ((dv * v).sum(-1) - L).detach()


def path_quantities(system, q0, v0, th, Q, dt, create_graph=True):
    """Cantidades variacionales de un camino Q de forma (B, N, d) que parte de (q0, v0).

    Devuelve un dict con:
      r       residuo de Euler–Lagrange discreto (∂S/∂q, con la condición inicial en r_0)
      pi      balance impulso–momento, cumsum(r)
      u       defecto de velocidad, M⁻¹π
      c       defecto de Picard, Φ(Q) − Q
      pN      momento final según la transformada de Legendre discreta, D2 L_d
      pN_int  momento final según el balance de impulso total
      E       energía en cada intervalo
      p0, w, mid   momento inicial, velocidades discretas y puntos medios
    """
    full = torch.cat([q0[:, None], Q], 1)
    mid = 0.5 * (full[:, 1:] + full[:, :-1])
    w = (full[:, 1:] - full[:, :-1]) / dt
    thN = th[:, None]
    dq, dv, L = dL(system, mid, w, thN, create_graph=create_graph)
    p0 = momentum(system, q0, v0, th)
    impulse = dt * dq
    pi = p0[:, None] + torch.cumsum(impulse, 1) - 0.5 * impulse - dv
    r = torch.cat([pi[:, :1], pi[:, 1:] - pi[:, :-1]], 1)
    u = apply_Minv(system, mid, w, thN, pi)
    c = dt * torch.cumsum(u, 1)
    return dict(r=r, pi=pi, u=u, c=c, pN=0.5 * impulse[:, -1] + dv[:, -1], pN_int=p0 + impulse.sum(1),
                E=(dv * w).sum(-1) - L, p0=p0, w=w, mid=mid)


def picard(system, q0, v0, th, Q, dt, n=1):
    """n iteraciones de punto fijo Q ← Φ(Q) = Q + c(Q). No usa ninguna red."""
    for _ in range(n):
        Q = (Q + path_quantities(system, q0, v0, th, Q.detach(), dt, create_graph=False)["c"]).detach()
    return Q


def free_flight(q0, v0, dt, N):
    """Camino sin fuerzas, q0 + v0·t."""
    t = dt * torch.arange(1, N + 1, device=q0.device, dtype=q0.dtype)[None, :, None]
    return q0[:, None] + t * v0[:, None]
