"""Redes que mapean (q0, v0, θ) a la trayectoria discreta q_1..q_N."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.a, self.b = nn.Linear(width, width), nn.Linear(width, width)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)

    def forward(self, x):
        return x + self.b(F.silu(self.a(F.silu(x))))


class TrajNet(nn.Module):
    """MLP residual con la condición inicial impuesta por construcción:

        q_k = q0 + t_k v0 + ½ t_k² · a_scale · NN_k(entradas)

    La red sólo aprende la «aceleración media hasta t_k», que es suave y de orden 1. La capa de
    salida empieza en cero, así que la predicción inicial es el vuelo libre. Una salida adicional
    estima la velocidad al final de la ventana, para poder encadenar ventanas sin evaluar fuerzas.
    """

    def __init__(self, system, N=None, T=None, width=256, depth=3, hard_ic=True):
        super().__init__()
        self.sys, self.N, self.T = system, N or system.N, T or system.T
        self.dt = self.T / self.N
        self.hard_ic = hard_ic
        self.inp = nn.Linear(system.n_features(), width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.out = nn.Linear(width, (self.N + 1) * system.d)
        if hard_ic:
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer("t", self.dt * torch.arange(1, self.N + 1, dtype=torch.float32))

    def forward(self, q0, v0, th, return_v=False):
        h = self.inp(self.sys.features(q0, v0, th))
        for block in self.blocks:
            h = block(h)
        out = self.out(F.silu(h)).view(-1, self.N + 1, self.sys.d)
        nodes, v_out = out[:, :-1], out[:, -1]
        q_scale, _, a_scale = self.sys.scales()
        t = self.t.to(out)[None, :, None]
        if self.hard_ic:
            Q = q0[:, None] + t * v0[:, None] + 0.5 * t ** 2 * a_scale * nodes
        else:
            Q = q0[:, None] + q_scale * nodes
        return (Q, v0 + self.T * a_scale * v_out) if return_v else Q


class OrigNet(nn.Module):
    """Arquitectura del proyecto original: entradas y salidas sin normalizar, con LayerNorm."""

    def __init__(self, system, N=None, T=None):
        super().__init__()
        self.sys, self.N, self.T = system, N or system.N, T or system.T
        self.dt = self.T / self.N
        self.net = nn.Sequential(
            nn.Linear(2 * system.d + system.P, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, system.d * self.N))

    def forward(self, q0, v0, th):
        out = self.net(torch.cat([q0, v0, th], -1))
        return out.view(-1, self.sys.d, self.N).transpose(1, 2)
