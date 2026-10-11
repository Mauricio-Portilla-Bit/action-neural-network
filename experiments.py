"""Experimentos del proyecto.

    python experiments.py --list
    python experiments.py NOMBRE [clave=valor ...]      # p. ej.  python experiments.py pendulum steps=5000

Cada corrida escribe results/NOMBRE.json (configuración, historial y métricas), results/NOMBRE.pt
(pesos) y results/NOMBRE.log.
"""
import json
import os
import sys
import time

import torch

import varnet as V

OUT = os.environ.get("VARNET_OUT", "results")


def config(**overrides):
    cfg = dict(
        system="projectile", sys_kw={},     # sistema y argumentos de su constructor
        N=None, T=None,                     # ventana; None = la del sistema
        model="traj", width=512, depth=4,   # "traj" (TrajNet) u "orig" (red del proyecto original)
        loss="pos", beta=1.0,               # ver varnet.losses; "sgK" = Picard con semigradiente
        steps=20000, batch=8192, lr=5e-4, clip=1.0, warmup=1000, orig_opt=False,
        dtype="f32", seed=0,
        eval_every=500, n_test=2000,
        rollout_K=0,                        # ventanas encadenadas en la evaluación final (0 = ninguna)
        load=None,                          # nombre de otra corrida de la que cargar pesos
    )
    cfg.update(overrides)
    return cfg


EXPERIMENTS = {}

# 1. Tiro parabólico: el proyecto original y qué cambia al sustituir cada pieza.
small = dict(batch=2048, eval_every=250)
small_new = dict(small, width=256, depth=3, lr=2e-3, clip=None, warmup=300)
EXPERIMENTS["proj_orig"] = config(model="orig", loss="orig", orig_opt=True, steps=500000, batch=2048, eval_every=2500)
EXPERIMENTS["proj_orignet_pos"] = config(model="orig", loss="pos", orig_opt=True, **small)
EXPERIMENTS["proj_traj_raw"] = config(loss="raw", **small_new)
EXPERIMENTS["proj_traj_pos"] = config(loss="pos", **small_new)

# 2. Péndulo: misma red y mismo presupuesto para cada pérdida, variando la resolución temporal.
for N in (25, 50, 100, 200):
    for name, loss, beta in (("raw", "raw", 1.0), ("mom", "mom", 1.0), ("pos", "pos", 1.0),
                             ("posmom", "posmom", 1.0), ("posmom01", "posmom", 0.01), ("sg1", "sg1", 1.0)):
        EXPERIMENTS[f"pendN{N}_{name}"] = config(system="pendulum", N=N, loss=loss, beta=beta, steps=10000,
                                                 batch=4096, width=256, depth=3, lr=2e-3, clip=None, warmup=300,
                                                 eval_every=250)

# 3. Sistemas no lineales. Ninguna variante del gradiente fue robusta en todos (ver README).
EXPERIMENTS["pendulum"] = config(system="pendulum", loss="sg1", steps=60000, rollout_K=30)
EXPERIMENTS["henon_heiles"] = config(system="henon_heiles", loss="sg1", steps=60000, rollout_K=25)
EXPERIMENTS["double_pendulum"] = config(system="double_pendulum", loss="sg1", steps=60000, rollout_K=40)
EXPERIMENTS["magnetic"] = config(system="magnetic", loss="pos", steps=60000, rollout_K=40)
EXPERIMENTS["double_pendulum_full_gradient"] = config(system="double_pendulum", loss="pos", steps=15000)
EXPERIMENTS["magnetic_long_window"] = config(system="magnetic", T=1.0, N=50, loss="pos", steps=60000, rollout_K=30)

# 4. Operador universal: entrenamiento con potenciales aleatorios y evaluación en potenciales nunca vistos.
universal = dict(system="universal", batch=8192, width=1024, lr=3e-4, n_test=1000)
EXPERIMENTS["universal"] = config(**universal, depth=6, steps=100000, eval_every=2000)
EXPERIMENTS["universal_xl"] = config(**universal, depth=8, steps=150000, eval_every=5000)
for potential in ("henon", "eggcrate", "gauss3", "kepler", "duffing"):
    for tag, source in (("uni", "universal"), ("unixl", "universal_xl")):
        EXPERIMENTS[f"{tag}_{potential}"] = dict(EXPERIMENTS[source], sys_kw=dict(potential=potential), steps=0,
                                                 load=source, rollout_K=20)


def build(cfg, device, dtype):
    system = V.SYSTEMS[cfg["system"]](**cfg["sys_kw"])
    N, T = cfg["N"] or system.N, cfg["T"] or system.T
    if cfg["model"] == "orig":
        model = V.OrigNet(system, N=N, T=T)
    else:
        model = V.TrajNet(system, N=N, T=T, width=cfg["width"], depth=cfg["depth"])
    return system, model.to(device, dtype)


def rmse(a, b):
    return (a - b).pow(2).sum(-1).mean().sqrt().item()


def never_drop(x):
    """Una trayectoria que diverge cuenta como error infinito; no se descarta de las estadísticas."""
    return torch.nan_to_num(x, nan=float("inf"), posinf=float("inf"))


@torch.no_grad()
def final_evaluation(system, model, testset, device, dtype):
    """Métricas finales: error por ventana, defecto de Picard contra error real, corrector y rollouts."""
    q0, v0, th = (testset[k].to(device, dtype) for k in ("q0", "v0", "th"))
    N, K = model.N, testset["K"]
    q_vi, q_rk = testset["q_vi"], testset["q_rk"]
    Q = model(q0, v0, th)
    defect = V.path_quantities(system, q0, v0, th, Q, model.dt, create_graph=False)["c"]
    Qc = Q.double().cpu()
    err_vi, err_rk = (Qc - q_vi).norm(dim=-1), (Qc - q_rk[:, :N]).norm(dim=-1)
    res = dict(
        rmse_vi=rmse(Qc, q_vi), rmse_rk=rmse(Qc, q_rk[:, :N]),
        rmse_disc=rmse(q_vi, q_rk[:, :N]),                       # error propio de la discretización
        q_scale=float(system.scales()[0]),
        max_err_vi=err_vi.max(1).values.tolist(),
        max_defect=defect.double().cpu().norm(dim=-1).max(1).values.tolist(),
        err_vi_t=err_vi.pow(2).mean(0).sqrt().tolist(), err_rk_t=err_rk.pow(2).mean(0).sqrt().tolist(),
    )

    def picard_curve(Qi, n=12):
        errors = [rmse(Qi.double().cpu(), q_vi)]
        for _ in range(n):
            Qi = V.picard(system, q0, v0, th, Qi, model.dt)
            errors.append(rmse(Qi.double().cpu(), q_vi))
        return errors

    res["picard_nn"] = picard_curve(Q)
    res["picard_free"] = picard_curve(V.free_flight(q0, v0, model.dt, N))
    if K <= 1:
        return res

    quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    stride = max(1, N // 10)
    E0 = V.energy(system, q0, v0, th).double().cpu()
    res["roll_t"] = (model.dt * torch.arange(1, K * N + 1))[::stride].tolist()
    res["roll_tw"] = (model.T * torch.arange(1, K + 1)).tolist()
    res["E_scale"] = E0.std().item()
    for mode in V.MODES:
        nodes, energies = V.rollout(model, system, q0, v0, th, K, mode=mode)
        nodes = nodes.double().cpu()
        err = never_drop((nodes - q_rk).norm(dim=-1))
        dE = never_drop((energies.double().cpu() - E0[:, None]).abs())
        res[f"roll_fail{mode}"] = (err[:, -1] > 1e3).double().mean().item()
        res[f"roll_err_q{mode}"] = torch.quantile(err, quantiles, dim=0)[:, ::stride].tolist()
        res[f"roll_dE_q{mode}"] = torch.quantile(dE, quantiles, dim=0).tolist()
        if mode == "nn":   # algunas trayectorias de ejemplo para las figuras
            res.update(ex_nn=nodes[:8, ::2].tolist(), ex_rk=q_rk[:8, ::2].tolist(),
                       ex_q0=testset["q0"][:8].tolist(), ex_v0=testset["v0"][:8].tolist(),
                       ex_th=testset["th"][:8].tolist())
    return res


def run(name, overrides=None):
    cfg = dict(EXPERIMENTS[name], **(overrides or {}))
    tag = cfg.pop("tag", name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if cfg["dtype"] == "f64" else torch.float32
    torch.manual_seed(cfg["seed"])
    system, model = build(cfg, device, dtype)
    os.makedirs(OUT, exist_ok=True)
    logfile = open(f"{OUT}/{tag}.log", "a")

    def log(line):
        print(line); logfile.write(line + "\n"); logfile.flush()

    log(f"### {tag}  {json.dumps(cfg)}  parámetros={sum(p.numel() for p in model.parameters()):,}")
    t0 = time.time()
    testset = V.make_testset(system, cfg["n_test"], model.N, model.T, K=max(1, cfg["rollout_K"]), device=device)
    log(f"referencias listas en {time.time() - t0:.0f}s")
    if cfg["load"]:
        model.load_state_dict(torch.load(f"{OUT}/{cfg['load']}.pt", map_location=device))
    hist = dict(step=[], time=[0.0], loss=[], e_vi=[], e_rk=[])
    if cfg["steps"] > 0:
        hist = V.train(system, model, loss=cfg["loss"], steps=cfg["steps"], batch=cfg["batch"], lr=cfg["lr"],
                       warmup=cfg["warmup"], clip=cfg["clip"], beta=cfg["beta"], orig_opt=cfg["orig_opt"],
                       device=device, dtype=dtype, testset=testset, eval_every=cfg["eval_every"],
                       seed=cfg["seed"], log=log, ckpt=f"{OUT}/{tag}.pt")
        torch.save(model.state_dict(), f"{OUT}/{tag}.pt")
    res = final_evaluation(system, model, testset, device, dtype)
    log(f"FINAL rmse_vi={res['rmse_vi']:.3e} rmse_rk={res['rmse_rk']:.3e} disc={res['rmse_disc']:.3e}")
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(dict(cfg=cfg, hist=hist, res=res, train_time=hist["time"][-1],
                       gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu"), f)


def parse_overrides(args):
    overrides = {}
    for arg in args:
        key, value = arg.split("=", 1)
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return overrides


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "--list"):
        print(__doc__)
        print("Experimentos:\n  " + "\n  ".join(EXPERIMENTS))
        sys.exit(0)
    run(sys.argv[1], parse_overrides(sys.argv[2:]))
