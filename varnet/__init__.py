"""varnet: redes entrenadas con el principio de acción discreto, sin datos.

Una red mapea (q0, v0, θ) a la trayectoria discreta q_1..q_N. La física entra únicamente por el
Lagrangiano L(q, v; θ); la pérdida es el residuo de Euler–Lagrange discreto medido en la norma de
Picard. Ver `mechanics` para las ecuaciones.
"""
from .losses import LOSSES, loss_fn
from .mechanics import (apply_Minv, dL, energy, free_flight, mass_matrix, momentum, path_quantities, picard,
                        velocity_from_momentum)
from .models import OrigNet, TrajNet
from .reference import rk4_reference, vi_reference
from .rollout import MODES, rollout, window, wrap_angles
from .systems import DoublePendulum, HenonHeiles, Magnetic, Pendulum, Projectile, System
from .training import evaluate, make_testset, train
from .universal import POTENTIALS, Universal

SYSTEMS = {s.name: s for s in (Projectile, Pendulum, HenonHeiles, DoublePendulum, Magnetic, Universal)}
