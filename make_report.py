"""Genera el reporte interactivo docs/index.html a partir de results/*.json.

    python make_report.py [--fragment RUTA]

El reporte es un único HTML autocontenido: la plantilla `report/body.html`, los estilos y el
código de las gráficas (`report/*.css|js`) y los datos de las figuras, todo incrustado.
`--fragment` escribe además el contenido sin la envoltura <html>, para publicarlo en un
servicio que añade la suya.
"""
import json
import math
import os
import sys

import numpy as np
import torch

import varnet as V

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SYSTEMS = [
    ("pendulum", "Péndulo", "θ₀∈[−π,π], θ̇₀∈[−4,4], ω²∈[0.25,4]"),
    ("henon_heiles", "Hénon–Heiles", "E < 1/6 (régimen caótico)"),
    ("double_pendulum", "Péndulo doble", "θ∈[−π,π]², θ̇∈[−2.5,2.5]², μ∈[0.5,2]"),
    ("magnetic", "Carga en campo B no uniforme", "B_z = B(1+β sin x sin y), B∈[−3,3], β∈[0,0.8], k∈[0,1]"),
]
POTENTIALS = [   # clave, nombre, tipo, caja de dibujo (x0, x1, y0, y1)
    ("henon", "Hénon–Heiles", "polinomio cúbico", (-1.0, 1.0, -0.8, 1.2)),
    ("eggcrate", "Red de cosenos", "periódico", (-6, 6, -6, 6)),
    ("gauss3", "Tres pozos gaussianos", "dispersión", (-3.5, 3.5, -3.5, 3.5)),
    ("kepler", "Kepler suavizado", "−0.3/√(r²+¼)", (-3.5, 3.5, -3.5, 3.5)),
    ("duffing", "Doble pozo acoplado", "polinomio cuártico", (-1.8, 1.8, -1.8, 1.8)),
]
PROJECTILE_RUNS = [
    ("proj_orig", "Original (red, pérdida y optimizador de main.ipynb)"),
    ("proj_orignet_pos", "Red y optimizador originales + pérdida de Picard"),
    ("proj_traj_raw", "Red nueva + pérdida cruda (sin energía)"),
    ("proj_traj_pos", "Red nueva + pérdida de Picard"),
]
ROLLOUT_MODES = ("nn", "int", "1", "2")


# ── utilidades ──────────────────────────────────────────────────────────────
def load(name):
    path = os.path.join(RESULTS, name + ".json")
    return json.load(open(path)) if os.path.exists(path) else None


def rounded(a, decimals):
    return np.round(np.asarray(a, dtype=float), decimals).tolist()


def sig(values, digits=3):
    """Redondea a cifras significativas; los no finitos pasan a None (JSON válido)."""
    return [float(f"{v:.{digits}g}") if math.isfinite(v) else None for v in np.asarray(values, dtype=float).ravel()]


def thin(steps, values, n=160):
    """Submuestrea una curva de entrenamiento con espaciado logarítmico."""
    idx = np.unique(np.round(np.geomspace(1, len(steps), min(n, len(steps)))).astype(int) - 1)
    return dict(x=[steps[i] for i in idx], y=sig([values[i] for i in idx]))


def first_step_below(hist, threshold):
    return next((s for s, e in zip(hist["step"], hist["e_vi"]) if e <= threshold), None)


def sci(v, decimals=1):
    """1.2e-3 -> 1.2·10⁻³"""
    if v is None or not math.isfinite(v):
        return "—"
    exponent = math.floor(math.log10(abs(v)))
    superscript = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{v / 10 ** exponent:.{decimals}f}·10{superscript}"


def thousands(v):
    return f"{v:,.0f}".replace(",", " ")


def table_row(*cells):
    """Celdas (texto, es_numérica) -> <tr>."""
    return "<tr>" + "".join(f"<td class='n'>{c}</td>" if numeric else f"<td>{c}</td>" for c, numeric in cells) + "</tr>"


# ── datos de las figuras ────────────────────────────────────────────────────
def condition_numbers(sizes=(25, 50, 100, 200, 400)):
    """Número de condición del Hessiano de cada pérdida para la partícula libre."""
    out = dict(N=list(sizes), raw=[], mom=[], pos=[])
    for N in sizes:
        A = np.eye(N) - 2 * np.eye(N, k=-1) + np.eye(N, k=-2)   # segunda diferencia causal
        C = np.tril(np.ones((N, N)))                            # suma acumulada
        for key, operator in (("raw", A), ("mom", C @ A), ("pos", C @ C @ A)):
            out[key].append(np.linalg.cond(operator) ** 2)
    return {k: (v if k == "N" else sig(v)) for k, v in out.items()}


def projectile_data():
    out = {}
    for key, _ in PROJECTILE_RUNS:
        run = load(key)
        if run:
            out[key] = dict(thin(run["hist"]["step"], run["hist"]["e_vi"]), final=run["res"]["rmse_vi"],
                            steps=run["hist"]["step"][-1], time=run["train_time"])
    return out


def ablation_data():
    final, curves = {}, {}
    for loss in ("raw", "mom", "posmom", "posmom01", "pos", "sg1"):
        runs = [(N, load(f"pendN{N}_{loss}")) for N in (25, 50, 100, 200)]
        runs = [(N, r) for N, r in runs if r]
        if runs:
            final[loss] = dict(x=[N for N, _ in runs], y=sig([r["res"]["rmse_vi"] for _, r in runs]))
    for loss in ("raw", "mom", "pos"):
        run = load(f"pendN100_{loss}")
        if run:
            curves[loss] = thin(run["hist"]["step"], run["hist"]["e_vi"], 80)
    return final, curves


def rollout_fields(res):
    out = dict(t=rounded(res["roll_t"], 3))
    for mode in ROLLOUT_MODES:
        out["roll_" + mode] = [sig(q) for q in res[f"roll_err_q{mode}"]]
        out["dE_" + mode] = [sig(q) for q in res[f"roll_dE_q{mode}"]]
        out["fail_" + mode] = res.get(f"roll_fail{mode}", 0.0)
    return out


def common_fields(res, n_points):
    return dict(rmse_vi=res["rmse_vi"], picard_nn=sig(res["picard_nn"]), picard_free=sig(res["picard_free"]),
                cert_x=sig(res["max_defect"][:n_points]), cert_y=sig(res["max_err_vi"][:n_points]))


def systems_data(example_horizon=12.0):
    out = {}
    for key, label, domain in SYSTEMS:
        run = load(key)
        if not run:
            continue
        res, cfg = run["res"], run["cfg"]
        N, window = len(res["err_vi_t"]), res["roll_tw"][0]
        example_dt = 2 * window / N                      # los ejemplos se guardaron cada 2 nodos
        n_example = int(example_horizon / example_dt)
        out[key] = dict(label=label, dom=domain, N=N, loss=cfg["loss"], steps=cfg["steps"], qs=res["q_scale"],
                        disc=res["rmse_disc"], tw=rounded(res["roll_tw"], 3), ex_dt=example_dt,
                        ex_nn=rounded(np.array(res["ex_nn"])[:4, :n_example], 3),
                        ex_rk=rounded(np.array(res["ex_rk"])[:4, :n_example], 3),
                        **common_fields(res, 500), **rollout_fields(res))
    return out


def universal_data(example_horizon=8.0):
    out = {}
    for tag, source in (("uni", "universal"), ("unixl", "universal_xl")):
        base = load(source)
        if not base:
            continue
        model = dict(steps=base["cfg"]["steps"], width=base["cfg"]["width"], depth=base["cfg"]["depth"],
                     rmse_random=base["res"]["rmse_vi"], pots={})
        for key, label, kind, box in POTENTIALS:
            run = load(f"{tag}_{key}")
            if not run:
                continue
            res = run["res"]
            n_example = int(example_horizon / (2 * 1.0 / len(res["err_vi_t"])))
            start = np.array(res["ex_q0"])[:5, None, :]
            with_start = lambda ex: rounded(np.concatenate([start, np.array(ex)[:5, :n_example]], 1), 3)
            model["pots"][key] = dict(label=label, kind=kind, box=box, ex_nn=with_start(res["ex_nn"]),
                                      ex_rk=with_start(res["ex_rk"]), **common_fields(res, 300), **rollout_fields(res))
        out[tag] = model
    return out


def potential_contours(key, box, n=90, levels=9):
    """Curvas de nivel de un potencial de prueba, como listas de polilíneas."""
    import contourpy
    x, y = np.linspace(box[0], box[1], n), np.linspace(box[2], box[3], n)
    grid = torch.tensor(np.stack(np.meshgrid(x, y), -1))
    Z = V.POTENTIALS[key]().V(grid, torch.zeros(1, 0)).numpy()
    generator = contourpy.contour_generator(x, y, Z)
    out = []
    for level in np.linspace(np.quantile(Z, 0.02), np.quantile(Z, 0.9), levels + 2)[1:-1]:
        segments = [rounded(seg[::2], 2) for seg in generator.lines(level) if len(seg) > 4]
        if segments:
            out.append(segments)
    return out


def gradient_comparison():
    """Péndulo doble: gradiente completo contra semigradiente, sobre los mismos pasos."""
    full, semi = load("double_pendulum_full_gradient"), load("double_pendulum")
    if not (full and semi):
        return None
    last = full["hist"]["step"][-1]
    n = sum(1 for s in semi["hist"]["step"] if s <= last)
    return dict(pos=thin(full["hist"]["step"], full["hist"]["e_vi"], 60),
                sg=thin(semi["hist"]["step"][:n], semi["hist"]["e_vi"][:n], 60),
                steps=last, full_final=full["res"]["rmse_vi"], semi_final=semi["hist"]["e_vi"][n - 1])


# ── fragmentos de texto que dependen de los datos ───────────────────────────
def projectile_rows(proj):
    target = proj["proj_orig"]["final"]
    rows = ""
    for key, label in PROJECTILE_RUNS:
        if key in proj:
            reached = first_step_below(load(key)["hist"], target)
            rows += table_row((label, False), (thousands(proj[key]["steps"]), True), (sci(proj[key]["final"]), True),
                              (thousands(reached) if reached else "no lo alcanza", True))
    return rows


def system_rows(systems):
    rows = ""
    for s in systems.values():
        rows += table_row(
            (f"<b>{s['label']}</b><br><span class='dim'>{s['dom']}</span>", False),
            (f"{s['tw'][0]:g} / {s['N']}", True),
            ("semigrad." if s["loss"].startswith("sg") else "completo", False),
            (sci(s["rmse_vi"]), True), (sci(s["rmse_vi"] / s["qs"]), True), (sci(s["disc"]), True),
            (sci(s["picard_nn"][1]), True), (sci(s["picard_nn"][2]), True), (thousands(s["steps"]), True))
    return rows


def universal_rows(model):
    rows = ""
    for p in model["pots"].values():
        mid = len(p["t"]) // 2 - 1                       # t = 10, la mitad del horizonte
        rows += table_row((f"<b>{p['label']}</b><br><span class='dim'>{p['kind']}</span>", False),
                          (sci(p["rmse_vi"]), True),
                          *((sci(p["roll_" + mode][1][mid]), True) for mode in ROLLOUT_MODES),
                          (sci(p["dE_1"][1][-1]), True))
    return rows


def gradient_section(cmp):
    if not cmp:
        return ""
    return ("<h3>Gradiente completo contra semigradiente</h3><p>El péndulo doble tiene fuerzas cuadráticas en la "
            "velocidad y matriz de masa dependiente de la configuración. "
            f"Tras {thousands(cmp['steps'])} pasos el gradiente completo queda en {sci(cmp['full_final'])} rad y el "
            f"semigradiente en {sci(cmp['semi_final'])} rad, con la misma red, los mismos datos y la misma tasa de "
            "aprendizaje. Con una ventana más larga (\\(T=1\\), prueba local) el gradiente completo bajó la pérdida "
            "sin bajar el error.</p><div class='fig'><div id='f_dp'></div><p class='cap'>Péndulo doble: RMSE por "
            "ventana (rad) durante el entrenamiento.</p></div>")


def scaling_section(universal):
    if not ("uni" in universal and "unixl" in universal):
        return ""
    a, b = universal["uni"], universal["unixl"]
    name = lambda m: f"{m['width']}×{m['depth']}, {thousands(m['steps'])} pasos"
    rows = table_row(("Campos aleatorios (distribución de entrenamiento)", False),
                     (sci(a["rmse_random"]), True), (sci(b["rmse_random"]), True))
    for key, p in b["pots"].items():
        if key in a["pots"]:
            rows += table_row((p["label"], False), (sci(a["pots"][key]["rmse_vi"]), True), (sci(p["rmse_vi"]), True))
    return ("<h3>¿Ayuda escalar el modelo?</h3>"
            f"<p>Dos corridas del operador universal: {name(a)} y {name(b)}.</p>"
            "<div class='tablewrap'><table><thead><tr><th>RMSE por ventana</th>"
            f"<th class='n'>{name(a)}</th><th class='n'>{name(b)}</th></tr></thead><tbody>{rows}</tbody></table></div>"
            "<p>Más profundidad y 50% más pasos mejoran el error entre 5% y 40% según el potencial. La ganancia es "
            "real pero modesta: con este diseño el límite no parece ser la capacidad de la red, sino la resolución "
            "del parche y la cola de potenciales difíciles.</p>")


def long_window_sentence(systems):
    long = load("magnetic_long_window")
    if not (long and "magnetic" in systems):
        return ""
    return (f" El sistema magnético con \\(T=1\\) quedó en {sci(long['res']['rmse_vi'])} por ventana; "
            f"con \\(T=0.5\\), en {sci(systems['magnetic']['rmse_vi'])}.")


# ── ensamblado ──────────────────────────────────────────────────────────────
def read(name):
    return open(os.path.join(HERE, "report", name), encoding="utf-8").read()


def build():
    proj, systems, universal = projectile_data(), systems_data(), universal_data()
    ablation, ablation_curves = ablation_data()
    cmp = gradient_comparison()
    data = dict(cond=condition_numbers(), proj=proj, abl=ablation, ablc=ablation_curves, systems=systems,
                uni=universal, contours={key: potential_contours(key, box) for key, _, _, box in POTENTIALS},
                bvp=load("bvp_action"), dp_cmp=cmp)
    best_tag = "unixl" if "unixl" in universal else "uni"
    best = universal[best_tag]
    original, new, swapped = proj["proj_orig"], load("proj_traj_pos"), load("proj_orignet_pos")
    speedup = original["steps"] / first_step_below(new["hist"], original["final"])
    speedup_loss_only = original["steps"] / first_step_below(swapped["hist"], original["final"])

    fields = {
        "SPEED": thousands(speedup), "SPEED2": thousands(speedup_loss_only),
        "CN4": sci(data["cond"]["raw"][0]),
        "ORIG_MIN": f"{original['time'] / 60:.0f}", "NEW_S": f"{proj['proj_traj_pos']['time']:.0f}",
        "PROJ_ROWS": projectile_rows(proj), "SYS_ROWS": system_rows(systems), "UNI_ROWS": universal_rows(best),
        "DP_CMP": gradient_section(cmp), "UNI_CMP": scaling_section(universal),
        "MAGT": long_window_sentence(systems),
        "DPFAIL": f"{100 * systems['double_pendulum']['fail_1']:.0f}%",
        "UNI_W": str(best["width"]), "UNI_D": str(best["depth"]), "UNI_STEPS": thousands(best["steps"]),
        "UNI_RAND": sci(best["rmse_random"]),
    }
    body = read("body.html")
    for key in sorted(fields, key=len, reverse=True):   # los nombres largos primero: SPEED2 antes que SPEED
        body = body.replace(f"__{key}__", fields[key])

    head = ('<title>Redes de Acción Discreta</title>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600'
            '&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
            f"<style>\n{read('style.css')}\n</style>\n")
    scripts = ('<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["$$","$$"]]},'
               'svg:{fontCache:"global"}};</script>\n'
               '<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js" async></script>\n'
               f"<script>\nconst DATA = {json.dumps(data, separators=(',', ':'))};\n"
               f"const UNI_TAG = '{best_tag}';\n</script>\n"
               f"<script>\n{read('charts.js')}\n</script>\n<script>\n{read('app.js')}\n</script>\n")
    main = f'<main class="page">\n{body}</main>\n'
    page = ('<!doctype html>\n<html lang="es"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"{head}</head><body>\n{main}{scripts}</body></html>\n")
    return page, head + main + scripts


if __name__ == "__main__":
    page, fragment = build()
    out = os.path.join(HERE, "docs", "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(page)
    print(f"docs/index.html  ({len(page) // 1024} KB)")
    if "--fragment" in sys.argv:
        open(sys.argv[sys.argv.index("--fragment") + 1], "w", encoding="utf-8").write(fragment)
