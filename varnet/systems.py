"""Sistemas mecánicos. Cada uno se define únicamente por su Lagrangiano L(q, v, θ)."""
import math

import torch


def uniform(B, lo, hi, device, dtype, gen=None):
    """Muestra B puntos uniformes en la caja [lo, hi]."""
    lo = torch.as_tensor(lo, dtype=dtype, device=device)
    hi = torch.as_tensor(hi, dtype=dtype, device=device)
    return lo + (hi - lo) * torch.rand(B, lo.numel(), device=device, dtype=dtype, generator=gen)


def henon_heiles_potential(q):
    x, y = q[..., 0], q[..., 1]
    return 0.5 * (x ** 2 + y ** 2) + x ** 2 * y - y ** 3 / 3.0


def inside_henon_heiles_well(q, E):
    """Región acotada del potencial: dentro del triángulo de escape y con E < 1/6."""
    return (E < 1 / 6) & (q[:, 1] < 0.5) & (q[:, 1] > -1 + math.sqrt(3) * q[:, 0].abs())


class System:
    """Interfaz de un sistema.

    Una subclase define `L` (que debe aceptar tensores `(..., d)` con cualquier forma de batch),
    `sample` (distribución de entrenamiento) y `scales`. Todo lo demás —fuerzas, momentos,
    matriz de masa, pérdida, integradores de referencia— se obtiene de `L` por autograd.
    """
    name = "base"
    d = 1               # grados de libertad
    P = 0               # número de parámetros θ del Lagrangiano
    T = 1.0             # duración de la ventana
    N = 50              # intervalos de tiempo por ventana
    unit_mass = True    # ∂²L/∂v² = I
    separable = False   # L = ½|v|² − V(q)
    angles = ()         # índices de las coordenadas periódicas

    def L(self, q, v, th):
        raise NotImplementedError

    def sample(self, B, device, dtype, gen=None):
        """-> (q0, v0, θ), cada uno de forma (B, ·)."""
        raise NotImplementedError

    def scales(self):
        """-> escalas típicas de posición, velocidad y aceleración."""
        raise NotImplementedError

    def features(self, q0, v0, th):
        """Entradas normalizadas de la red; los ángulos entran como (sin, cos)."""
        qs, vs, _ = self.scales()
        feats = []
        for i in range(self.d):
            qi = q0[..., i:i + 1]
            feats += [torch.sin(qi), torch.cos(qi)] if i in self.angles else [qi / qs]
        feats.append(v0 / vs)
        if self.P:
            feats.append(self.th_features(th))
        return torch.cat(feats, -1)

    def th_features(self, th):
        return th

    def n_features(self):
        q0, v0, th = self.sample(2, torch.device("cpu"), torch.float32)
        return self.features(q0, v0, th).shape[-1]


class Projectile(System):
    """Tiro parabólico, con el dominio del proyecto original (posiciones y velocidades en ±15)."""
    name, d, P, T, N = "projectile", 2, 0, 1.25, 25
    separable = True
    g = 9.81

    def __init__(self, box=15.0):
        self.box = box

    def L(self, q, v, th):
        return 0.5 * (v ** 2).sum(-1) - self.g * q[..., 1]

    def sample(self, B, device, dtype, gen=None):
        b = self.box
        q0 = uniform(B, [-b, -b], [b, b], device, dtype, gen)
        v0 = uniform(B, [-b, -b], [b, b], device, dtype, gen)
        return q0, v0, torch.zeros(B, 0, device=device, dtype=dtype)

    def scales(self):
        return self.box, self.box, self.g


class Pendulum(System):
    """L = ½θ'² + ω² cos θ, con θ = [ω²]. El dominio incluye libración y rotación."""
    name, d, P, T, N = "pendulum", 1, 1, 1.0, 50
    separable = True
    angles = (0,)

    def L(self, q, v, th):
        return 0.5 * v[..., 0] ** 2 + th[..., 0] * torch.cos(q[..., 0])

    def sample(self, B, device, dtype, gen=None):
        q0 = uniform(B, [-math.pi], [math.pi], device, dtype, gen)
        v0 = uniform(B, [-4.0], [4.0], device, dtype, gen)
        th = uniform(B, [0.25], [4.0], device, dtype, gen)
        return q0, v0, th

    def scales(self):
        return math.pi, 4.0, 4.0

    def th_features(self, th):
        return (th - 2.125) / 1.875


class HenonHeiles(System):
    """V = ½(x²+y²) + x²y − y³/3. Caótico al acercarse a la energía de escape E = 1/6."""
    name, d, P, T, N = "henon_heiles", 2, 0, 2.0, 100
    separable = True

    def L(self, q, v, th):
        return 0.5 * (v ** 2).sum(-1) - henon_heiles_potential(q)

    def sample(self, B, device, dtype, gen=None):
        qs, vs, n = [], [], 0
        while n < B:   # muestreo por rechazo dentro del pozo
            q = uniform(2 * B, [-0.6, -0.6], [0.6, 0.9], device, dtype, gen)
            v = uniform(2 * B, [-0.6, -0.6], [0.6, 0.6], device, dtype, gen)
            ok = inside_henon_heiles_well(q, 0.5 * (v ** 2).sum(-1) + henon_heiles_potential(q))
            qs.append(q[ok]); vs.append(v[ok]); n += int(ok.sum())
        return torch.cat(qs)[:B], torch.cat(vs)[:B], torch.zeros(B, 0, device=device, dtype=dtype)

    def scales(self):
        return 0.6, 0.6, 0.6


class DoublePendulum(System):
    """Péndulo doble con g = l₁ = l₂ = m₁ = 1 y θ = [μ = m₂/m₁].

    La matriz de masa depende de la configuración, así que L no es separable.
    """
    name, d, P, T, N = "double_pendulum", 2, 1, 0.5, 25
    unit_mass = False
    angles = (0, 1)

    def L(self, q, v, th):
        mu = th[..., 0]
        a, b = q[..., 0], q[..., 1]
        va, vb = v[..., 0], v[..., 1]
        return (0.5 * (1 + mu) * va ** 2 + 0.5 * mu * vb ** 2 + mu * va * vb * torch.cos(a - b)
                + (1 + mu) * torch.cos(a) + mu * torch.cos(b))

    def Minv(self, q, th, x):
        """M⁻¹x en forma cerrada. Es una optimización: sin este método se usa el Hessiano de L."""
        mu = th[..., 0]
        c = torch.cos(q[..., 0] - q[..., 1])
        det = mu * (1 + mu) - (mu * c) ** 2
        return torch.stack([(mu * x[..., 0] - mu * c * x[..., 1]) / det,
                            (-mu * c * x[..., 0] + (1 + mu) * x[..., 1]) / det], -1)

    def sample(self, B, device, dtype, gen=None):
        q0 = uniform(B, [-math.pi, -math.pi], [math.pi, math.pi], device, dtype, gen)
        v0 = uniform(B, [-2.5, -2.5], [2.5, 2.5], device, dtype, gen)
        th = uniform(B, [0.5], [2.0], device, dtype, gen)
        return q0, v0, th

    def scales(self):
        return math.pi, 2.5, 3.0

    def th_features(self, th):
        return (th - 1.25) / 0.75


class Magnetic(System):
    """Carga en un campo magnético no uniforme: L = ½|v|² + v·A(q) − φ(q).

    B_z = B(1 + β sin x sin y), con el gauge A = (0, B(x − β cos x sin y)), más una trampa
    armónica φ = ½k r². θ = [B, β, k]. El potencial depende de la velocidad.
    """
    name, d, P, T, N = "magnetic", 2, 3, 0.5, 25

    def L(self, q, v, th):
        Bz, beta, k = th[..., 0], th[..., 1], th[..., 2]
        x, y = q[..., 0], q[..., 1]
        Ay = Bz * (x - beta * torch.cos(x) * torch.sin(y))
        return 0.5 * (v ** 2).sum(-1) + v[..., 1] * Ay - 0.5 * k * (x ** 2 + y ** 2)

    def sample(self, B, device, dtype, gen=None):
        q0 = uniform(B, [-2, -2], [2, 2], device, dtype, gen)
        v0 = uniform(B, [-1, -1], [1, 1], device, dtype, gen)
        th = uniform(B, [-3.0, 0.0, 0.0], [3.0, 0.8, 1.0], device, dtype, gen)
        return q0, v0, th

    def scales(self):
        return 2.0, 1.0, 3.0

    def th_features(self, th):
        return torch.stack([th[..., 0] / 3.0, th[..., 1] / 0.4 - 1, 2 * th[..., 2] - 1.0], -1)
