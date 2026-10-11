"""Operador universal: una red para potenciales 2D arbitrarios.

La red no recibe parámetros ni fórmulas del potencial. Recibe la velocidad inicial y los valores
de V en una malla local centrada en la partícula.
"""
import math

import torch

from .systems import System, henon_heiles_potential, inside_henon_heiles_well, uniform


class RandomPotential:
    """Familia de entrenamiento: campo aleatorio suave + polinomio aleatorio de grado ≤ 4.

    Cada muestra del batch es un potencial distinto, descrito por θ:
    J ondas (k_x, k_y, fase, amplitud) y 14 coeficientes polinomiales.
    """
    J = 24
    P = 4 * J + 14

    def V(self, q, th):
        J = self.J
        x, y = q[..., 0:1], q[..., 1:2]
        kx, ky, ph, am = th[..., :J], th[..., J:2 * J], th[..., 2 * J:3 * J], th[..., 3 * J:4 * J]
        field = (am * torch.cos(x * kx + y * ky + ph)).sum(-1)
        c = th[..., 4 * J:]
        x, y = x[..., 0], y[..., 0]
        poly = (c[..., 0] * x + c[..., 1] * y
                + 0.5 * (c[..., 2] * x * x + 2 * c[..., 3] * x * y + c[..., 4] * y * y)
                + c[..., 5] * x ** 3 + c[..., 6] * x * x * y + c[..., 7] * x * y * y + c[..., 8] * y ** 3
                + c[..., 9] * x ** 4 + c[..., 10] * x ** 3 * y + c[..., 11] * x * x * y * y
                + c[..., 12] * x * y ** 3 + c[..., 13] * y ** 4)
        return field + poly

    def sample(self, B, device, dtype, gen=None):
        J = self.J
        rand = lambda *s: torch.rand(*s, device=device, dtype=dtype, generator=gen)
        randn = lambda *s: torch.randn(*s, device=device, dtype=dtype, generator=gen)
        ell = 0.7 + 1.8 * rand(B, 1)                       # longitud de correlación del campo
        k = randn(B, J, 2) / ell[:, :, None]
        phase = 2 * math.pi * rand(B, J)
        force_std = 1.2 * rand(B, 1) ** 1.5
        amp = force_std * ell * math.sqrt(2.0 / J) * torch.ones(B, J, device=device, dtype=dtype)
        grad = randn(B, 2) * rand(B, 1)
        # Hessiano simétrico con autovalores en [-2.5, 3], a veces casi nulo
        angle = math.pi * rand(B)
        shrink = rand(B)
        l1, l2 = (-2.5 + 5.5 * rand(B)) * shrink, (-2.5 + 5.5 * rand(B)) * shrink
        cs, sn = torch.cos(angle), torch.sin(angle)
        hxx, hyy, hxy = l1 * cs ** 2 + l2 * sn ** 2, l1 * sn ** 2 + l2 * cs ** 2, (l1 - l2) * cs * sn
        cubic = 0.6 * randn(B, 4) * rand(B, 1) ** 2
        quartic = 0.15 * randn(B, 5) * rand(B, 1) ** 2
        th = torch.cat([k[..., 0], k[..., 1], phase, amp, grad,
                        hxx[:, None], hxy[:, None], hyy[:, None], cubic, quartic], 1)
        # La familia es invariante por traslación, así que basta con partir del origen.
        q0 = torch.zeros(B, 2, device=device, dtype=dtype)
        v0 = uniform(B, [-1, -1], [1, 1], device, dtype, gen)
        return q0, v0, th


class NamedPotential:
    """Potencial de prueba fijo (sin parámetros), fuera de la distribución de entrenamiento."""
    P = 0
    box_q, box_v = 2.0, 0.6

    def V(self, q, th):
        raise NotImplementedError

    def accept(self, q, v, E):
        return torch.ones_like(E, dtype=torch.bool)

    def sample(self, B, device, dtype, gen=None):
        qs, vs, n = [], [], 0
        no_params = torch.zeros(1, 0, device=device, dtype=dtype)
        while n < B:
            q = uniform(2 * B, [-self.box_q] * 2, [self.box_q] * 2, device, dtype, gen)
            v = uniform(2 * B, [-self.box_v] * 2, [self.box_v] * 2, device, dtype, gen)
            ok = self.accept(q, v, 0.5 * (v ** 2).sum(-1) + self.V(q, no_params))
            qs.append(q[ok]); vs.append(v[ok]); n += int(ok.sum())
        return torch.cat(qs)[:B], torch.cat(vs)[:B], torch.zeros(B, 0, device=device, dtype=dtype)


class HenonPotential(NamedPotential):
    box_q, box_v = 0.9, 0.6

    def V(self, q, th):
        return henon_heiles_potential(q)

    def accept(self, q, v, E):
        return inside_henon_heiles_well(q, E)


class EggcratePotential(NamedPotential):
    box_q, box_v = math.pi, 0.6

    def V(self, q, th):
        x, y = q[..., 0], q[..., 1]
        return -0.2 * (torch.cos(x) + torch.cos(y)) + 0.1 * torch.cos(x + y)


class ThreeGaussiansPotential(NamedPotential):
    amplitudes = (-0.4, -0.3, 0.3)
    widths = (0.8, 0.6, 0.7)
    centers = ((-0.8, 0.3), (0.9, -0.6), (0.2, 1.0))

    def V(self, q, th):
        out = 0
        for a, s, c in zip(self.amplitudes, self.widths, self.centers):
            out = out + a * torch.exp(-((q[..., 0] - c[0]) ** 2 + (q[..., 1] - c[1]) ** 2) / (2 * s * s))
        return out


class SoftKeplerPotential(NamedPotential):
    box_q, box_v = 2.0, 0.5

    def V(self, q, th):
        return -0.3 / torch.sqrt((q ** 2).sum(-1) + 0.25)


class DoubleWellPotential(NamedPotential):
    box_q, box_v = 1.6, 0.9

    def V(self, q, th):
        x, y = q[..., 0], q[..., 1]
        return 0.25 * (x * x - 1) ** 2 + 0.5 * y * y + 0.25 * x * x * y * y

    def accept(self, q, v, E):
        return E < 0.4


POTENTIALS = dict(random=RandomPotential, henon=HenonPotential, eggcrate=EggcratePotential,
                  gauss3=ThreeGaussiansPotential, kepler=SoftKeplerPotential, duffing=DoubleWellPotential)


class Universal(System):
    """Partícula de masa unidad en un potencial 2D arbitrario.

    Simetrías impuestas por construcción en las entradas de la red:
      · traslación: la malla está centrada en q0, que no se le da a la red;
      · gauge: sólo entra V − V(q0);
      · energía: los valores se recortan en `vcap`, por encima de la energía cinética máxima,
        porque la partícula no puede llegar ahí.
    """
    name, d, T, N = "universal", 2, 1.0, 50
    separable = True
    R, G = 2.0, 13              # semiancho de la malla y puntos por lado
    vcap, vlow = 1.25, -3.0

    def __init__(self, potential="random"):
        self.pot = POTENTIALS[potential]()
        self.P = self.pot.P
        g = torch.linspace(-self.R, self.R, self.G)
        self.grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)

    def L(self, q, v, th):
        return 0.5 * (v ** 2).sum(-1) - self.pot.V(q, th)

    def sample(self, B, device, dtype, gen=None):
        return self.pot.sample(B, device, dtype, gen)

    def scales(self):
        return 1.0, 1.0, 2.0

    def features(self, q0, v0, th):
        points = q0[:, None, :] + self.grid.to(q0)[None]
        dV = self.pot.V(points, th[:, None]) - self.pot.V(q0, th)[:, None]
        return torch.cat([v0, dV.clamp(self.vlow, self.vcap)], -1)

    def n_features(self):
        return 2 + self.G ** 2
