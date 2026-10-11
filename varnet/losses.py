"""Funciones de coste. Todas se anulan en la misma trayectoria; difieren en la norma del residuo."""
from .mechanics import energy

LOSSES = ("pos", "mom", "posmom", "raw", "orig")


def loss_fn(system, pq, dt, kind, q0=None, v0=None, th=None, beta=1.0):
    """Pérdida a partir de `path_quantities`.

      pos     ‖c‖²   Picard: residuo integrado dos veces (la que usa el proyecto)
      mom     ‖π‖²   balance impulso–momento: residuo integrado una vez
      posmom  pos + beta · mom
      raw     ‖r‖²   residuo de Euler–Lagrange discreto sin integrar
      orig    pérdida del proyecto original: raw + conservación de energía, con sus pesos

    La variante de semigradiente de `pos` (`sgK`) vive en `training.train`, porque cambia cómo
    se deriva y no el valor de la pérdida.
    """
    qs = system.scales()[0]
    N = pq["c"].shape[1]
    T = dt * N
    if kind == "pos":
        return ((pq["c"] / qs) ** 2).mean()
    if kind == "mom":
        return ((pq["u"] * T / qs) ** 2).mean()
    if kind == "posmom":
        return ((pq["c"] / qs) ** 2).mean() + beta * ((pq["u"] * T / qs) ** 2).mean()
    if kind == "raw":
        return ((pq["r"] * T * T / (dt * qs)) ** 2).mean()
    if kind == "orig":
        r = pq["r"]
        stationarity = (r[:, 1:] ** 2).sum(-1).mean() / dt ** 2
        initial_momentum = (r[:, 0] ** 2).sum(-1).mean() * N
        E0 = energy(system, q0, v0, th)
        E = pq["E"][:, :-1]
        energy_term = ((E - E0[:, None]) ** 2).mean() + E.var(dim=1).mean()
        return 20.0 * stationarity + 100.0 * energy_term + 50.0 * initial_momentum
    raise ValueError(f"pérdida desconocida: {kind!r}")
