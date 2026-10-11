"""Inferencia: una ventana y composición de ventanas."""
import math

import torch

from .mechanics import energy, path_quantities, picard, velocity_from_momentum

MODES = ("nn", "int", 1, 2)


def wrap_angles(system, q):
    """Lleva las coordenadas angulares a [−π, π)."""
    if not system.angles:
        return q
    q = q.clone()
    for i in system.angles:
        q[..., i] = torch.remainder(q[..., i] + math.pi, 2 * math.pi) - math.pi
    return q


def window(model, system, q0, v0, th, mode="nn"):
    """Predice una ventana. Devuelve (nodos, v_N).

    mode:
      "nn"   todo sale de la red; cero evaluaciones de fuerza
      "int"  nodos de la red; v_N por balance de impulso sobre el camino predicho
      k ≥ 1  k correcciones de Picard y después balance de impulso
    """
    with torch.no_grad():
        Q, vN = model(q0, v0, th, return_v=True)
    if mode != "nn":
        Q = picard(system, q0, v0, th, Q, model.dt, 0 if mode == "int" else int(mode))
        pN = path_quantities(system, q0, v0, th, Q, model.dt, create_graph=False)["pN_int"]
        vN = velocity_from_momentum(system, Q[:, -1].detach(), pN.detach(), th)
    return Q.detach(), vN.detach()


def rollout(model, system, q0, v0, th, K, mode="nn"):
    """Encadena K ventanas (el sistema es autónomo, así que la misma red sirve para todas).

    Devuelve los nodos (B, K·N, d), con los ángulos sin envolver, y la energía al final de cada
    ventana (B, K).
    """
    nodes, energies, q, v = [], [], q0, v0
    for _ in range(K):
        q_wrapped = wrap_angles(system, q)
        Q, v = window(model, system, q_wrapped, v, th, mode)
        Q = Q + (q - q_wrapped)[:, None]
        nodes.append(Q)
        q = Q[:, -1]
        energies.append(energy(system, q, v, th))
    return torch.cat(nodes, 1), torch.stack(energies, 1)
