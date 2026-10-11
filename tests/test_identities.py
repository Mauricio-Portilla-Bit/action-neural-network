"""Identidades matemáticas en las que se apoya el proyecto.

    python -m pytest tests            (o, sin pytest:  python tests/test_identities.py)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import varnet as V

CPU, F64 = torch.device("cpu"), torch.float64


def setup(name, n=16, **kw):
    system = V.SYSTEMS[name](**kw)
    q0, v0, th = system.sample(n, CPU, F64, torch.Generator().manual_seed(0))
    dt = system.T / system.N
    q_vi, _ = V.vi_reference(system, q0, v0, th, dt, system.N)
    return system, q0, v0, th, dt, q_vi


def all_systems():
    return [(name, {}) for name in V.SYSTEMS if name != "universal"] + [("universal", dict(potential="henon"))]


def test_residual_vanishes_on_variational_integrator():
    """r, π y c se anulan en la misma trayectoria: la del integrador variacional."""
    for name, kw in all_systems():
        system, q0, v0, th, dt, q_vi = setup(name, **kw)
        pq = V.path_quantities(system, q0, v0, th, q_vi, dt, create_graph=False)
        for key in ("r", "pi", "c"):
            assert pq[key].abs().max() < 1e-9, (name, key)


def test_residual_telescopes_to_momentum_balance():
    """π = cumsum(r) para cualquier camino, no sólo para la solución."""
    for name, kw in all_systems():
        system, q0, v0, th, dt, q_vi = setup(name, **kw)
        Q = q_vi + 0.1 * torch.randn(q_vi.shape, dtype=F64, generator=torch.Generator().manual_seed(1))
        pq = V.path_quantities(system, q0, v0, th, Q, dt, create_graph=False)
        assert torch.allclose(torch.cumsum(pq["r"], 1), pq["pi"], atol=1e-10), name


def test_picard_defect_is_the_error_for_constant_forces():
    """Con fuerzas independientes de q, c = q* − q exactamente (pérdida de condición 1)."""
    system, q0, v0, th, dt, q_vi = setup("projectile")
    Q = q_vi + torch.randn(q_vi.shape, dtype=F64, generator=torch.Generator().manual_seed(1))
    c = V.path_quantities(system, q0, v0, th, Q, dt, create_graph=False)["c"]
    assert torch.allclose(c, q_vi - Q, atol=1e-10)


def test_projectile_matches_parabola_and_energy_bias():
    """El integrador de punto medio es exacto en los nodos, pero su energía discreta tiene un
    sesgo constante de −g²Δt²/8. Por eso penalizar (E − E₀)² desplaza el mínimo."""
    system, q0, v0, th, dt, q_vi = setup("projectile")
    t = dt * torch.arange(1, system.N + 1, dtype=F64)[None, :, None]
    gravity = torch.tensor([0.0, -system.g], dtype=F64)
    assert torch.allclose(q_vi, q0[:, None] + v0[:, None] * t + 0.5 * gravity * t ** 2, atol=1e-10)
    E = V.path_quantities(system, q0, v0, th, q_vi, dt, create_graph=False)["E"]
    bias = E - V.energy(system, q0, v0, th)[:, None]
    assert torch.allclose(bias, torch.full_like(bias, -system.g ** 2 * dt ** 2 / 8), atol=1e-9)


def test_condition_numbers():
    """Partícula libre: la pérdida cruda tiene condición ~N⁴; la de Picard, exactamente 1."""
    N = 25
    A = np.eye(N) - 2 * np.eye(N, k=-1) + np.eye(N, k=-2)      # segunda diferencia causal
    C = np.tril(np.ones((N, N)))                               # suma acumulada
    assert np.linalg.cond(A) ** 2 > 1e5
    assert abs(np.linalg.cond(C @ C @ A) - 1) < 1e-9


def test_picard_iteration_converges_from_free_flight():
    system, q0, v0, th, dt, q_vi = setup("pendulum")
    Q = V.picard(system, q0, v0, th, V.free_flight(q0, v0, dt, system.N), dt, n=12)
    assert (Q - q_vi).abs().max() < 1e-8


def test_variational_integrator_agrees_with_rk4():
    for name, kw in all_systems():
        system, q0, v0, th, dt, q_vi = setup(name, **kw)
        q_rk, _ = V.rk4_reference(system, q0, v0, th, dt, system.N)
        assert (q_vi - q_rk).abs().max() < 5e-3, name


def test_closed_form_mass_inverse_matches_autograd():
    system, q0, v0, th, dt, _ = setup("double_pendulum")
    x = torch.randn(q0.shape, dtype=F64, generator=torch.Generator().manual_seed(2))
    M = V.mass_matrix(system, q0, v0, th)
    assert torch.allclose(system.Minv(q0, th, x), torch.linalg.solve(M, x.unsqueeze(-1)).squeeze(-1), atol=1e-12)


def test_network_starts_at_free_flight():
    system = V.Pendulum()
    q0, v0, th = system.sample(8, CPU, torch.float32, torch.Generator().manual_seed(0))
    model = V.TrajNet(system, width=32, depth=1)
    assert torch.allclose(model(q0, v0, th), V.free_flight(q0, v0, model.dt, model.N), atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
