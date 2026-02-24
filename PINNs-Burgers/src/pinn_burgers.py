"""
pinn_burgers.py
---------------
PINN for 1D viscous Burgers' equation with clean directory structure, detailed
comments, benchmark comparison, and rich plots.

PDE: u_t + u * u_x = nu * u_xx,   x in [-1,1],  t in [0,1]
IC:  u(x,0) = -sin(pi x)
BC:  u(-1,t) = u(1,t) = 0
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ----------------------------
# Paths (save outside src/)
# ----------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parents[1]        # parent of src/
SRC_DIR = PROJECT_DIR / "src"
CKPT_DIR = PROJECT_DIR / "checkpoints"
OUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUT_DIR / "figs"
for d in (CKPT_DIR, OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Device & dtype
# ----------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32  # MPS prefers float32
torch.manual_seed(1234)

# ----------------------------
# Domain & physics
# ----------------------------
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX =  0.0, 1.0
NU = 0.01 / math.pi  # viscosity

# ----------------------------
# PINN hyperparameters (good defaults)
# ----------------------------
# Collocation counts per epoch (interior : IC : BC ~ 10 : 1 : 1)
N_F  = 40_000
N_IC = 3_000
N_BC = 3_000

# Network capacity (MLP with tanh activations)
WIDTH = 50   # try 50 → 64 if interior remains rough
DEPTH = 8    # try 8 → 10 for tougher cases

# Training schedule
LR = 1e-3
EPOCHS_ADAM = 5_000
PRINT_EVERY = 500

# ----------------------------
# Evaluation grid (for visualization & benchmark comparison)
# ----------------------------
NX_EVAL = 256
NT_EVAL = 101

# ----------------------------
# Utilities
# ----------------------------
def normalize(x, xmin, xmax):
    """Map x from [xmin, xmax] to [-1, 1] for numerically stable NN inputs."""
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

def ic_u(x):
    """Initial condition: u(x,0) = -sin(pi x). Tensor in, tensor out."""
    return -torch.sin(math.pi * x)

# ----------------------------
# PINN model
# ----------------------------
class MLP(nn.Module):
    """
    Smooth MLP mapping (x,t) -> u, using tanh so derivatives (u_x, u_xx, u_t)
    computed by autodiff are well behaved.
    """
    def __init__(self, in_dim=2, out_dim=1, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):  # add (depth-1) hidden blocks
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]  # final linear: no activation
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        """Xavier init for Linear layers; zeros for biases."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t):
        """
        Forward expects column tensors x and t with shape [N,1], already normalized.
        Returns u with shape [N,1].
        """
        xt = torch.cat([x, t], dim=1)  # [N,2]
        return self.net(xt)

# ----------------------------
# Sampling functions
# ----------------------------
def sample_collocation(n):
    """Random interior points (x,t) ~ Uniform on domain rectangle."""
    x = torch.rand(n, 1, device=DEVICE, dtype=DTYPE) * (X_MAX - X_MIN) + X_MIN
    t = torch.rand(n, 1, device=DEVICE, dtype=DTYPE) * (T_MAX - T_MIN) + T_MIN
    return x, t

def sample_ic(n):
    """Random x along t=0 line, with target u0(x)."""
    x = torch.rand(n, 1, device=DEVICE, dtype=DTYPE) * (X_MAX - X_MIN) + X_MIN
    t = torch.zeros_like(x) + T_MIN
    u = ic_u(x)
    return x, t, u

def sample_bc(n):
    """Random t along left/right boundaries; targets are zero (Dirichlet)."""
    t = torch.rand(n, 1, device=DEVICE, dtype=DTYPE) * (T_MAX - T_MIN) + T_MIN
    x_left  = torch.zeros_like(t) + X_MIN
    x_right = torch.zeros_like(t) + X_MAX
    u_left  = torch.zeros_like(t)  # u(-1,t)=0
    u_right = torch.zeros_like(t)  # u( 1,t)=0
    return x_left, t, u_left, x_right, t, u_right

# ----------------------------
# Physics residual via autodiff
# ----------------------------
def pde_residual(model, x, t):
    """
    Compute Burgers residual r = u_t + u * u_x - nu * u_xx at given (x,t).
    Autodiff handles chain rule and higher derivatives.
    """
    x.requires_grad_(True)
    t.requires_grad_(True)

    xn = normalize(x, X_MIN, X_MAX)
    tn = normalize(t, T_MIN, T_MAX)

    u = model(xn, tn)  # [N,1]

    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True)[0]

    r = du_dt + u * du_dx - NU * d2u_dx2
    return r

# ----------------------------
# Optional: batched interior loss (stabilizes memory/variance)
# ----------------------------
def collocation_loss_batched(model, total_N=N_F, batch=8_000):
    """
    Estimate mean(r^2) by averaging over mini-batches of collocation points.
    Same objective, often smoother gradients and lower memory.
    """
    batches = max(1, total_N // batch)
    acc = 0.0
    for _ in range(batches):
        x_f, t_f = sample_collocation(batch)
        r = pde_residual(model, x_f, t_f)
        acc += (r**2).mean()
    return acc / batches

# ----------------------------
# Training
# ----------------------------
def train():
    model = MLP().to(DEVICE).to(DTYPE)
    adam = torch.optim.Adam(model.parameters(), lr=LR)

    best_residual_mean = float("inf")
    best_path = CKPT_DIR / "burgers_pinn_best.pt"

    # Adam phase: stochastic collocation each epoch
    for epoch in range(1, EPOCHS_ADAM + 1):

        # Interior physics loss (choose one of the two lines)
        # loss_f = (pde_residual(model, *sample_collocation(N_F))**2).mean()
        loss_f = collocation_loss_batched(model, total_N=N_F, batch=min(8_000, N_F))

        # Initial-condition loss (t = 0)
        x_ic, t_ic, u_ic = sample_ic(N_IC)
        u_pred_ic = model(normalize(x_ic, X_MIN, X_MAX), normalize(t_ic, T_MIN, T_MAX))
        loss_ic = ((u_pred_ic - u_ic)**2).mean()

        # Boundary-condition loss (x = -1 and x = +1)
        x_l, t_l, u_l, x_r, t_r, u_r = sample_bc(N_BC)
        u_pred_l = model(normalize(x_l, X_MIN, X_MAX), normalize(t_l, T_MIN, T_MAX))
        u_pred_r = model(normalize(x_r, X_MIN, X_MAX), normalize(t_r, T_MIN, T_MAX))
        loss_bc = (u_pred_l**2).mean() + (u_pred_r**2).mean()

        # Two-phase weighting: emphasize IC/BC early, then increase PDE weight
        if epoch < int(0.5 * EPOCHS_ADAM):
            lam_f, lam_ic, lam_bc = 1.0, 200.0, 200.0
        else:
            lam_f, lam_ic, lam_bc = 2.0, 100.0, 100.0

        loss = lam_f * loss_f + lam_ic * loss_ic + lam_bc * loss_bc

        adam.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        adam.step()

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS_ADAM:
            # Log weighted terms to see balance
            wf, wic, wbc = lam_f*loss_f.item(), lam_ic*loss_ic.item(), lam_bc*loss_bc.item()
            print(f"[Adam {epoch:5d}] total={loss.item():.3e} | wf={wf:.3e} ic={wic:.3e} bc={wbc:.3e}")

            # Light-weight "validation": residual mean on a small grid
            stats = residual_stats(model, nx=96, nt=48)
            res_mean = stats["mean"]
            # Save best-by-residual checkpoint
            if res_mean < best_residual_mean:
                best_residual_mean = res_mean
                torch.save(model.state_dict(), best_path)
                print(f"  Saved new BEST checkpoint to {best_path} (res_mean={res_mean:.3e})")

    # Save last Adam checkpoint
    last_adam_path = CKPT_DIR / "burgers_pinn_adam_last.pt"
    torch.save(model.state_dict(), last_adam_path)
    print(f"Saved Adam-last checkpoint to {last_adam_path}")

    # L-BFGS refinement: freeze samples for a deterministic closure
    Xf_LB, Tf_LB = sample_collocation(N_F)
    Xic_LB, Tic_LB, Uic_LB = sample_ic(N_IC)
    Xl_LB, Tl_LB, Ul_LB, Xr_LB, Tr_LB, Ur_LB = sample_bc(N_BC)

    def closure():
        adam.zero_grad(set_to_none=True)
        r = pde_residual(model, Xf_LB, Tf_LB)

        uic = model(normalize(Xic_LB, X_MIN, X_MAX), normalize(Tic_LB, T_MIN, T_MAX))
        ul  = model(normalize(Xl_LB,  X_MIN, X_MAX), normalize(Tl_LB,  T_MIN, T_MAX))
        ur  = model(normalize(Xr_LB,  X_MIN, X_MAX), normalize(Tr_LB,  T_MIN, T_MAX))

        loss_f  = (r**2).mean()
        loss_ic = ((uic - Uic_LB)**2).mean()
        loss_bc = (ul**2).mean() + (ur**2).mean()

        # Use the "late" weights for refinement
        loss = 2.0 * loss_f + 100.0 * loss_ic + 100.0 * loss_bc
        loss.backward()
        return loss

    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0,
        max_iter=500, max_eval=500,
        history_size=50, line_search_fn="strong_wolfe"
    )
    t0 = time.time()
    final_loss = lbfgs.step(closure)
    print(f"[L-BFGS] final_loss={final_loss.item():.3e} (elapsed {time.time()-t0:.1f}s)")

    # Save final refined checkpoint
    final_path = CKPT_DIR / "burgers_pinn_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final checkpoint to {final_path}")

    return model

# ----------------------------
# Evaluation & saving
# ----------------------------
@torch.no_grad()
def eval_and_save(model, nx=NX_EVAL, nt=NT_EVAL, out_path=OUT_DIR / "burgers_pinn_results.pt"):
    """Evaluate u(x,t) on a dense grid (for plotting / comparison) and save."""
    model.eval()
    x = torch.linspace(X_MIN, X_MAX, nx, device=DEVICE, dtype=DTYPE).view(-1,1)
    t = torch.linspace(T_MIN, T_MAX, nt, device=DEVICE, dtype=DTYPE).view(-1,1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    Xf, Tf = X.reshape(-1,1), T.reshape(-1,1)
    U = model(normalize(Xf, X_MIN, X_MAX), normalize(Tf, T_MIN, T_MAX)).reshape(nx, nt).cpu()
    torch.save({"x": X.cpu(), "t": T.cpu(), "u": U}, out_path)
    print(f"Saved PINN field to {out_path}")

# ----------------------------
# Residual & constraint metrics (physics-only checks)
# ----------------------------
def residual_stats(model, nx=128, nt=64):
    """
    Compute mean/median/max(|r|) on a moderate grid.
    Note: we need gradients here, so do NOT wrap with no_grad().
    """
    model.eval()
    x = torch.linspace(X_MIN, X_MAX, nx, device=DEVICE, dtype=DTYPE).view(-1,1)
    t = torch.linspace(T_MIN, T_MAX, nt, device=DEVICE, dtype=DTYPE).view(-1,1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    Xf, Tf = X.reshape(-1,1), T.reshape(-1,1)
    r = pde_residual(model, Xf, Tf).detach().abs().view(nx, nt)  # detach after computing residual
    return {"mean": r.mean().item(), "median": r.median().item(), "max": r.max().item()}

def ic_bc_rmse_from_saved(path=OUT_DIR / "burgers_pinn_results.pt"):
    """RMSE on IC (t=0) and both BC edges from saved tensor file."""
    D = torch.load(path)
    X, T, U = D["x"].numpy(), D["t"].numpy(), D["u"].numpy()  # [nx,nt] each
    x = X[:, 0]
    ic_rmse = float(np.sqrt(np.mean((U[:, 0] - (-np.sin(np.pi * x)))**2)))
    bc_left  = float(np.sqrt(np.mean(U[0, :]**2)))
    bc_right = float(np.sqrt(np.mean(U[-1,:]**2)))
    return ic_rmse, bc_left, bc_right

# ----------------------------
# Reference FD solver (benchmark)
# ----------------------------
def burgers_fd_reference(nx=NX_EVAL, nt_cap=100_000, store_times=NT_EVAL):
    """
    Explicit finite-difference reference:
    - Upwind for convection, centered second difference for diffusion.
    - Adaptive time step for CFL and diffusion stability.
    Returns X, T, U_ref with shapes matching eval grid.
    """
    # Spatial grid
    x = np.linspace(X_MIN, X_MAX, nx)
    dx = x[1] - x[0]
    dx2 = dx * dx

    # Target times to store (match eval grid)
    t_targets = np.linspace(T_MIN, T_MAX, store_times)

    # Initial condition
    u = -np.sin(np.pi * x)
    u[0]  = 0.0  # enforce Dirichlet at boundaries
    u[-1] = 0.0

    U_snaps = np.zeros((nx, store_times), dtype=np.float64)
    U_snaps[:, 0] = u.copy()

    t = 0.0
    k = 1  # next target index to fill
    steps = 0

    while k < store_times and steps < nt_cap:
        # Stability-limited dt
        maxu = max(1e-6, float(np.max(np.abs(u))))
        dt_conv = dx / maxu                     # CFL for advection
        dt_diff = 0.5 * dx2 / NU                # explicit diffusion
        dt = 0.4 * min(dt_conv, dt_diff)        # safety factor

        # Compute derivatives on interior (vectorized)
        ux  = np.zeros_like(u)
        uxx = np.zeros_like(u)

        # Upwind for u_x
        m_pos = u[1:-1] >= 0.0
        ux[1:-1][m_pos]  = (u[1:-1][m_pos] - u[0:-2][m_pos]) / dx
        ux[1:-1][~m_pos] = (u[2:][~m_pos] - u[1:-1][~m_pos]) / dx

        # Centered second derivative for diffusion
        uxx[1:-1] = (u[2:] - 2.0*u[1:-1] + u[0:-2]) / dx2

        # Explicit Euler update
        un = u - dt * u * ux + NU * dt * uxx

        # Enforce Dirichlet BCs
        un[0]  = 0.0
        un[-1] = 0.0

        # Advance time
        t_next = t + dt

        # Save snapshot(s) if we crossed target times
        while k < store_times and t <= t_targets[k] <= t_next:
            alpha = (t_targets[k] - t) / (t_next - t)  # linear in time between u and un
            U_snaps[:, k] = (1 - alpha) * u + alpha * un
            k += 1

        # Prepare next step
        u = un
        t = t_next
        steps += 1

    # Fill any remaining targets with the last state
    while k < store_times:
        U_snaps[:, k] = u
        k += 1

    X = np.tile(x[:, None], (1, store_times))
    T = np.tile(t_targets[None, :], (nx, 1))
    return X, T, U_snaps

def compare_with_benchmark(pinn_path=OUT_DIR / "burgers_pinn_results.pt"):
    """
    Load PINN field, run FD benchmark on the same grid, and print relative L2 error.
    Also returns (X, T, U_pinn, U_ref, rel_L2).
    """
    D = torch.load(pinn_path)
    Xp, Tp, Up = D["x"].numpy(), D["t"].numpy(), D["u"].numpy()  # [nx, nt]
    Xr, Tr, Ur = burgers_fd_reference(nx=Xp.shape[0], store_times=Tp.shape[1])

    # Relative L2 error (Frobenius norm over space-time)
    num = np.linalg.norm(Up - Ur)
    den = np.linalg.norm(Ur) + 1e-12
    rel_L2 = float(num / den)

    print(f"[Benchmark] Relative L2 error (PINN vs FD): {rel_L2:.3e}")
    return Xp, Tp, Up, Ur, rel_L2

# ----------------------------
# Plots
# ----------------------------
def quick_plots(X, T, U_pinn, U_ref=None, out_dir=FIG_DIR):
    """
    Heatmaps: PINN, FD, absolute error; plus line-cuts and IC/BC checks,
    residual histogram, and energy vs time.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap: PINN
    plt.figure(figsize=(6,4))
    plt.pcolormesh(T, X, U_pinn, shading="auto")
    plt.xlabel("t"); plt.ylabel("x"); plt.title("PINN u(x,t)")
    plt.colorbar(label="u")
    plt.tight_layout()
    plt.savefig(out_dir / "pinn_heatmap.png", dpi=160)
    plt.close()

    if U_ref is not None:
        # Heatmap: FD
        plt.figure(figsize=(6,4))
        plt.pcolormesh(T, X, U_ref, shading="auto")
        plt.xlabel("t"); plt.ylabel("x"); plt.title("FD reference u(x,t)")
        plt.colorbar(label="u")
        plt.tight_layout()
        plt.savefig(out_dir / "fd_heatmap.png", dpi=160)
        plt.close()

        # Heatmap: |error|
        plt.figure(figsize=(6,4))
        plt.pcolormesh(T, X, np.abs(U_pinn - U_ref), shading="auto")
        plt.xlabel("t"); plt.ylabel("x"); plt.title("|PINN - FD|")
        plt.colorbar(label="|error|")
        plt.tight_layout()
        plt.savefig(out_dir / "abs_error.png", dpi=160)
        plt.close()

        # Line cuts: overlay PINN vs FD at selected times
        times = [0.25, 0.5, 0.75, 1.0]
        for tt in times:
            j = int(np.argmin(np.abs(T[0,:] - tt)))
            plt.plot(X[:,0], U_pinn[:,j], label=f"PINN t={T[0,j]:.2f}")
            plt.plot(X[:,0], U_ref[:,j],  linestyle="--", label=f"FD t={T[0,j]:.2f}")
        plt.xlabel("x"); plt.ylabel("u")
        plt.title("Line cuts: PINN vs FD")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "linecuts.png", dpi=160)
        plt.close()

    # IC and BC checks from U_pinn alone
    # IC at t=0
    x = X[:,0]
    ic_true = -np.sin(np.pi * x)
    plt.figure(figsize=(6,4))
    plt.plot(x, U_pinn[:,0], label="PINN t=0")
    plt.plot(x, ic_true, linestyle="--", label="IC true")
    # BCs across time (should be ~0)
    plt.plot(T[0,:], U_pinn[0,:], alpha=0.6, label="BC left (x=-1)")
    plt.plot(T[-1,:], U_pinn[-1,:], alpha=0.6, label="BC right (x=+1)")
    plt.xlabel("x / t"); plt.ylabel("u")
    plt.title("IC and BC checks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ic_bc.png", dpi=160)
    plt.close()

    # Residual histogram (compute fresh at moderate grid)
    # Note: requires grads ⇒ do not decorate with no_grad
    model_tmp = None  # not available here; skip residual histogram in this function

def residual_histogram(model, nx=128, nt=64, out_path=FIG_DIR / "residual_hist.png"):
    """Standalone residual histogram plot."""
    import matplotlib.pyplot as plt
    x = torch.linspace(X_MIN, X_MAX, nx, device=DEVICE, dtype=DTYPE).view(-1,1)
    t = torch.linspace(T_MIN, T_MAX, nt, device=DEVICE, dtype=DTYPE).view(-1,1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    Xf, Tf = X.reshape(-1,1), T.reshape(-1,1)
    r = pde_residual(model, Xf, Tf).detach().cpu().numpy().ravel()
    plt.figure(figsize=(6,4))
    plt.hist(np.abs(r), bins=60)
    plt.xlabel("|residual|"); plt.ylabel("count")
    plt.title("Residual magnitude histogram (PINN)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def energy_vs_time(U, X, out_path=FIG_DIR / "energy_vs_time.png"):
    """Plot kinetic energy E(t) = 0.5 * ∫ u^2 dx (should decrease with time for viscous flow)."""
    import matplotlib.pyplot as plt
    dx = (X[-1,0] - X[0,0]) / (X.shape[0] - 1)
    E = 0.5 * np.trapz(U**2, dx=dx, axis=0)
    t = np.linspace(T_MIN, T_MAX, U.shape[1])
    plt.figure(figsize=(6,4))
    plt.plot(t, E)
    plt.xlabel("t"); plt.ylabel("E(t)")
    plt.title("Energy vs time (PINN)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and save checkpoints.")
    parser.add_argument("--eval", action="store_true", help="Evaluate model to grid and save tensors.")
    parser.add_argument("--benchmark", action="store_true", help="Run FD reference and compare.")
    parser.add_argument("--plots", action="store_true", help="Generate plots in outputs/figs.")
    args = parser.parse_args()

    run_all = not (args.train or args.eval or args.benchmark or args.plots)

    # Stage 1: train (or load best/final if training is skipped)
    model = None
    if args.train or run_all:
        model = train()
    else:
        model = MLP().to(DEVICE).to(DTYPE)
        # Prefer final checkpoint if present; else best; else Adam-last
        if (CKPT_DIR / "burgers_pinn_final.pt").exists():
            model.load_state_dict(torch.load(CKPT_DIR / "burgers_pinn_final.pt", map_location=DEVICE))
        elif (CKPT_DIR / "burgers_pinn_best.pt").exists():
            model.load_state_dict(torch.load(CKPT_DIR / "burgers_pinn_best.pt", map_location=DEVICE))
        elif (CKPT_DIR / "burgers_pinn_adam_last.pt").exists():
            model.load_state_dict(torch.load(CKPT_DIR / "burgers_pinn_adam_last.pt", map_location=DEVICE))
        else:
            raise FileNotFoundError("No checkpoint found. Run with --train first.")
    model.eval()

    # Stage 2: evaluate to dense grid
    if args.eval or run_all:
        eval_and_save(model)

    # Stage 3: physics checks + benchmark comparison (+ save metrics)
    metrics = {}
    stats = residual_stats(model)
    metrics["residual_mean"] = stats["mean"]
    metrics["residual_median"] = stats["median"]
    metrics["residual_max"] = stats["max"]
    ic_rmse, bcL, bcR = ic_bc_rmse_from_saved(OUT_DIR / "burgers_pinn_results.pt")
    metrics["ic_rmse"] = ic_rmse
    metrics["bc_rmse_left"] = bcL
    metrics["bc_rmse_right"] = bcR

    if args.benchmark or run_all:
        X, T, U_pinn, U_ref, rel = compare_with_benchmark(OUT_DIR / "burgers_pinn_results.pt")
        metrics["rel_L2_vs_FD"] = rel
    else:
        # Load PINN field for plots below
        D = torch.load(OUT_DIR / "burgers_pinn_results.pt")
        X, T, U_pinn = D["x"].numpy(), D["t"].numpy(), D["u"].numpy()
        U_ref = None

    # Save metrics JSON
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to", OUT_DIR / "metrics.json")

    # Stage 4: plots
    if args.plots or run_all:
        quick_plots(X, T, U_pinn, U_ref)
        residual_histogram(model, out_path=FIG_DIR / "residual_hist.png")
        energy_vs_time(U_pinn, X, out_path=FIG_DIR / "energy_vs_time.png")
        print("Saved figures to", FIG_DIR)

if __name__ == "__main__":
    main()