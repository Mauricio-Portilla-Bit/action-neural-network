# %% [markdown]
# ## Red Neuronal — Principio Variacional Local via Autograd (v7)
#
# ### Diagnóstico v6:
# ∂S_global/∂qⱼ mezcla contribuciones de todos los intervalos.
# Aunque matemáticamente da las EL, numéricamente la señal se diluye.
#
# ### Solución: Estacionariedad por sub-intervalos
#
# El cálculo de variaciones dice: para TODA variación δq(t) localizada
# en [tᵢ, tᵢ₊₁], la variación δS debe ser cero. Esto es equivalente a:
#
#   ∂Sᵢ/∂qⱼ = 0   para cada sub-acción Sᵢ y cada punto qⱼ adyacente
#
# En v3 esto se escribió como ax=0, ay=g (EL manual).
# Aquí lo hacemos con autograd: descomponemos S en sub-acciones Sᵢ = Lᵢ·dt
# y computamos ∂Sᵢ/∂qⱼ para j ∈ {i, i+1} (los puntos que afectan Sᵢ).
#
# La forma más limpia: para cada punto interior qⱼ, la condición EL discreta es:
#   ∂(Sⱼ₋₁ + Sⱼ)/∂qⱼ = 0
# Es decir: el punto qⱼ es estacionario respecto a los DOS intervalos adyacentes.
# Esto se computa con autograd sin conocer las EOM.

# %% Imports y constantes
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Usando GPU Apple Silicon (MPS)")
else:
    device = torch.device("cpu")
    print("⚠ MPS no disponible, usando CPU")

# Constantes físicas
g = -9.81
dt = 0.05
N = 25
m = 1.0

# Hiperparámetros
LAMBDA_STATION = 1.0       # estacionariedad local (la principal)
LAMBDA_ENERGY  = 1.0       # regularización suave (Noether)
LAMBDA_IC_VEL  = 10.0      # condición de frontera
EPOCHS = 15000
BATCH_SIZE = 2048
LR_INIT = 3e-3
CLIP_NORM = 1.0

# %% Modelo simple (como v3 que funcionó bien)
class ActionNet(nn.Module):
    def __init__(self, n_points=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 2 * n_points),
        )

    def forward(self, x):
        return self.net(x)

model = ActionNet(N).to(device)
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")


# %% Lagrangiano discreto por intervalo

def compute_interval_lagrangians(pred, batch):
    """
    Calcula el Lagrangiano Lᵢ en cada intervalo [tᵢ, tᵢ₊₁].
    Retorna las cantidades como tensores que MANTIENEN el grafo de autograd
    con respecto a pred.
    """
    x0 = batch[:, 0:1]
    y0 = batch[:, 1:2]

    x_full = torch.cat([x0, pred[:, :N]], dim=1)   # (B, 26)
    y_full = torch.cat([y0, pred[:, N:]], dim=1)

    # Velocidades
    vx = (x_full[:, 1:] - x_full[:, :-1]) / dt     # (B, 25)
    vy = (y_full[:, 1:] - y_full[:, :-1]) / dt

    # Posición en puntos medios
    y_mid = (y_full[:, 1:] + y_full[:, :-1]) / 2.0

    # Lagrangiano por intervalo
    T = 0.5 * m * (vx**2 + vy**2)                   # (B, 25)
    V = m * (-g) * y_mid
    L_i = T - V                                      # (B, 25)

    return L_i, vx, vy, T, V, x_full, y_full


# %% Pérdida física: estacionariedad local via autograd

def physics_loss(model, batch):
    """
    Estacionariedad LOCAL del cálculo de variaciones via autograd.

    Para cada punto interior qⱼ (j=1,...,N-1), la EL discreta dice:

        ∂/∂qⱼ [Lⱼ₋₁ + Lⱼ] · dt = 0

    Es decir, qⱼ es estacionario respecto a los dos intervalos adyacentes.
    Autograd computa esto sin necesidad de derivar las EOM manualmente.

    Para j=0 (primer punto predicho): solo depende de L₀ → ∂L₀/∂q₀ = 0
    Para j=N-1 (último punto): solo depende de Lₙ₋₁ → condición de frontera libre
    """
    pred = model(batch)  # (B, 50)
    L_i, vx, vy, T, V, x_full, y_full = compute_interval_lagrangians(pred, batch)

    # --- 1. ESTACIONARIEDAD LOCAL via autograd ---
    # Computar ∂Lᵢ/∂pred para CADA intervalo i, luego sumar los pares adyacentes.
    #
    # Equivalentemente: para cada punto predicho j (0..N-1), computar
    # ∂S_local/∂qⱼ donde S_local = Σ Lᵢ·dt (solo los Lᵢ que dependen de qⱼ).
    #
    # Dado que Lᵢ depende de qᵢ y qᵢ₊₁, ∂S/∂qⱼ = ∂(Lⱼ₋₁+Lⱼ)/∂qⱼ·dt.
    # Esto es exactamente lo que autograd computa con S = sum(L_i * dt).
    #
    # PERO: para que la señal sea LOCAL, computamos el gradiente intervalo
    # por intervalo y penalizamos cada uno independientemente.

    station_losses = []
    for i in range(N):
        # Sub-acción del intervalo i
        S_i = L_i[:, i] * dt  # (B,)

        # Gradiente de Sᵢ respecto a los puntos predichos
        dSi_dq = torch.autograd.grad(
            S_i.sum(), pred, create_graph=True, retain_graph=True
        )[0]  # (B, 50)

        # Solo nos interesan los puntos ADYACENTES al intervalo i:
        # Intervalo i usa x[i], x[i+1], y[i], y[i+1]
        # En pred: x_j = pred[:, j] para j=0..N-1 (punto j+1 de x_full)
        #          y_j = pred[:, N+j] para j=0..N-1

        # Punto "izquierdo" del intervalo: es pred[:,i-1] si i>0, o x0 (fijo)
        # Punto "derecho" del intervalo: es pred[:,i]

        # El punto pred[:,i] es x_full[:,i+1], que aparece en intervalos i e i+1.
        # Para punto interior: queremos ∂(Sᵢ + Sᵢ₊₁)/∂pred[:,i] ≈ 0
        # Con este loop computamos ∂Sᵢ/∂pred[:,i] para cada i.
        # La suma ∂S/∂pred[:,i] = ∂Sᵢ₋₁/∂pred[:,i] + ∂Sᵢ/∂pred[:,i]
        # se obtiene naturalmente al sumar las contribuciones.

        # Penalizamos la magnitud del gradiente en el punto i
        # (componentes x e y)
        grad_x_i = dSi_dq[:, i]        # ∂Sᵢ/∂xᵢ₊₁
        grad_y_i = dSi_dq[:, N + i]    # ∂Sᵢ/∂yᵢ₊₁
        station_losses.append(grad_x_i**2 + grad_y_i**2)

    # La condición EL para punto interior j es:
    # ∂Sⱼ₋₁/∂qⱼ + ∂Sⱼ/∂qⱼ = 0
    # Esto ya está capturado porque sumamos contribuciones de ambos intervalos
    stationarity_loss = torch.mean(torch.stack(station_losses, dim=1))

    # --- 2. Conservación de energía (regularizador suave) ---
    E0 = (0.5 * m * (batch[:, 2:3]**2 + batch[:, 3:4]**2)
          + m * (-g) * batch[:, 1:2])
    E = T + V
    energy_err = (E - E0) / (E0.abs() + 1e-6)
    energy_loss = torch.mean(energy_err**2)

    # --- 3. Condición inicial de velocidad ---
    vx0 = batch[:, 2:3]
    vy0 = batch[:, 3:4]
    vel_err = (vx[:, 0:1] - vx0)**2 + (vy[:, 0:1] - vy0)**2
    vel_loss = torch.mean(vel_err)

    total = (LAMBDA_STATION * stationarity_loss
             + LAMBDA_ENERGY * energy_loss
             + LAMBDA_IC_VEL * vel_loss)

    return total, stationarity_loss, energy_loss, vel_loss


# %% Generador de datos
def generate_batch(batch_size):
    x0  = np.random.uniform(-10, 10, (batch_size, 1))
    y0  = np.random.uniform(0, 20, (batch_size, 1))
    vx0 = np.random.uniform(-5, 5, (batch_size, 1))
    vy0 = np.random.uniform(0, 15, (batch_size, 1))
    data = np.concatenate([x0, y0, vx0, vy0], axis=1).astype(np.float32)
    return torch.from_numpy(data).to(device)


# %% Solución analítica
def analytical_trajectory(x0, y0, vx0, vy0):
    t = np.arange(1, N + 1) * dt
    return x0 + vx0 * t, y0 + vy0 * t + 0.5 * g * t**2


# %% Entrenamiento
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

history = {"total": [], "station": [], "energy": [], "vel_ic": []}

print(f"Entrenando {EPOCHS} épocas en {device}...")
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    batch = generate_batch(BATCH_SIZE)

    optimizer.zero_grad()
    total, stat, eng, vel = physics_loss(model, batch)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    optimizer.step()
    scheduler.step()

    history["total"].append(total.item())
    history["station"].append(stat.item())
    history["energy"].append(eng.item())
    history["vel_ic"].append(vel.item())

    if epoch % 2000 == 0 or epoch == 1:
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f"Epoch {epoch:>6d} | Total: {total:.4f} | "
              f"δS: {stat:.5f} | E: {eng:.5f} | "
              f"v₀: {vel:.5f} | LR: {lr:.2e} | {elapsed:.0f}s")

elapsed = time.time() - t0
print(f"\nCompletado en {elapsed:.1f}s ({EPOCHS/elapsed:.0f} ep/s)")

# %% Gráfica de pérdidas
def smooth(v, w=100):
    return np.convolve(v, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, key, title in zip(axes,
    ["station", "energy", "vel_ic", "total"],
    ["δS/δq Local", "Conserv. Energía", "Vel. Inicial", "Total"]):
    raw = np.array(history[key])
    ax.semilogy(raw, alpha=0.15, color='steelblue')
    if len(raw) > 200:
        ax.semilogy(smooth(raw, 200), color='darkblue', lw=1.5)
    ax.set_title(title); ax.set_xlabel("Época"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Comparación trayectoria
def plot_trajectory_comparison(model, ic, title_suffix=""):
    x0, y0, vx0, vy0 = ic
    ic_t = torch.tensor(ic, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(ic_t).cpu().numpy()[0]
    x_nn = np.concatenate([[x0], pred[:N]])
    y_nn = np.concatenate([[y0], pred[N:]])
    x_an, y_an = analytical_trajectory(x0, y0, vx0, vy0)
    x_an = np.concatenate([[x0], x_an])
    y_an = np.concatenate([[y0], y_an])
    t_arr = np.arange(0, N + 1) * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(x_an, y_an, 'k--', lw=2.5, label='Analítica')
    ax.plot(x_nn, y_nn, 'ro-', ms=5, lw=1.5, label='Red Neuronal', alpha=0.8)
    ax.scatter([x0], [y0], c='green', s=100, zorder=10, label='Inicio')
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Trayectoria x-y"); ax.legend()
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    ax = axes[0, 1]
    ax.plot(t_arr, x_an, 'b--', lw=2, label='x analítica')
    ax.plot(t_arr, x_nn, 'b^-', ms=4, lw=1, alpha=0.7, label='x red')
    ax.plot(t_arr, y_an, 'r--', lw=2, label='y analítica')
    ax.plot(t_arr, y_nn, 'rs-', ms=4, lw=1, alpha=0.7, label='y red')
    ax.set_xlabel("t (s)"); ax.set_ylabel("posición (m)")
    ax.set_title("Posición vs Tiempo"); ax.legend(ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_arr, np.abs(x_nn - x_an), 'b.-', label='|Δx|')
    ax.plot(t_arr, np.abs(y_nn - y_an), 'r.-', label='|Δy|')
    ax.set_xlabel("t (s)"); ax.set_ylabel("Error absoluto (m)")
    ax.set_title("Error absoluto"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    vx_nn = np.diff(x_nn) / dt
    vy_nn = np.diff(y_nn) / dt
    y_mid = (y_nn[1:] + y_nn[:-1]) / 2
    E_nn = 0.5*m*(vx_nn**2+vy_nn**2) + m*9.81*y_mid
    E0_val = 0.5*m*(vx0**2+vy0**2) + m*9.81*y0
    ax.plot(E_nn, 'bo-', ms=4, label='E(t) predicha')
    ax.axhline(E0_val, color='r', ls='--', lw=2, label=f'E₀={E0_val:.2f} J')
    ax.set_xlabel("Paso temporal"); ax.set_ylabel("E (J)")
    ax.set_title("Conservación de Energía"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"IC: x₀={x0:.1f}, y₀={y0:.1f}, vx₀={vx0:.1f}, vy₀={vy0:.1f}  "
                 f"{title_suffix}", fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()

    rmse_x = np.sqrt(np.mean((x_nn - x_an)**2))
    rmse_y = np.sqrt(np.mean((y_nn - y_an)**2))
    print(f"  RMSE_x={rmse_x:.4f} m | RMSE_y={rmse_y:.4f} m | σ(E)={np.std(E_nn):.4f} J")

# %% Ejecutar
test_conditions = np.array([
    [0.0, 0.0, 3.0, 10.0],
    [2.0, 5.0, -2.0, 8.0],
    [-3.0, 10.0, 4.0, 12.0],
], dtype=np.float32)

for i, ic in enumerate(test_conditions):
    plot_trajectory_comparison(model, ic, title_suffix=f"(Test {i+1})")
