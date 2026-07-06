#!/usr/bin/env python3
"""
Corrected grid convergence study for wave equation FDM.
T=10 (five full periods) as stated in the paper.
Also generates Fig 6 with Nx=100,200,400 curves.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json

# ----- solver -----
def solve_wave(L=1.0, c=1.0, T=10.0, Nx=100, sigma=0.9):
    dx = L / Nx
    dt = sigma * dx / c
    Nt = int(T / dt)
    x = np.linspace(0, L, Nx+1)
    u_exact_fn = lambda x, t: np.sin(np.pi*x/L) * np.cos(np.pi*c*t/L)
    u_prev = np.sin(np.pi*x/L).copy()
    u_curr = u_prev.copy()
    u_curr[0] = u_curr[-1] = 0
    u_prev[0] = u_prev[-1] = 0
    sigma2 = sigma**2
    for n in range(Nt):
        u_next = np.zeros(Nx+1)
        u_next[1:-1] = (2*(1-sigma2)*u_curr[1:-1] +
                       sigma2*(u_curr[2:]+u_curr[:-2]) -
                       u_prev[1:-1])
        u_next[0] = u_next[-1] = 0
        u_prev, u_curr = u_curr, u_next
    u_exact = u_exact_fn(x, T)
    error = u_curr - u_exact
    L2 = np.sqrt(np.sum(error**2) / Nx)
    Linf = np.max(np.abs(error))
    return L2, Linf, x, u_curr, u_exact

# Paths
output_dir = r"D:\飞书协作群1资料\14_波动方程FDM教学\02_研究执行\data"
fig_dir = r"D:\飞书协作群1资料\14_波动方程FDM教学\03_论文输出\figures"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

L, c = 1.0, 1.0
T_conv = 10.0  # five full periods
Nx_list = [20, 40, 80, 160, 320, 640, 1280]
sigma_vals = [0.5, 0.9, 1.0]

results = {}
print(f"Convergence study: T={T_conv}, L={L}, c={c}")
print(f"{'sigma':>6} {'Nx':>5} {'L2_err':>12} {'Linf_err':>12} {'rate':>6}")
print("-"*45)

for sigma in sigma_vals:
    L2_vals = []
    Linf_vals = []
    for Nx in Nx_list:
        L2, Linf, _, _, _ = solve_wave(L, c, T=T_conv, Nx=Nx, sigma=sigma)
        L2_vals.append(L2)
        Linf_vals.append(Linf)
    rates = []
    for i in range(len(L2_vals)-1):
        r = np.log(L2_vals[i]/L2_vals[i+1]) / np.log(2)
        rates.append(round(r, 2))
    results[str(sigma)] = {
        'Nx': Nx_list, 'L2': L2_vals, 'Linf': Linf_vals, 'rate_L2': rates
    }
    for i, Nx in enumerate(Nx_list):
        r_str = f"{rates[i-1]:.2f}" if i > 0 else "---"
        print(f"{sigma:>5.1f} {Nx:>5} {L2_vals[i]:>12.3e} {Linf_vals[i]:>12.3e} {r_str:>6}")

# Save data
data_path = os.path.join(output_dir, 'convergence_data_T10.json')
with open(data_path, 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"\n✅ Data saved to {data_path}")

# Fig 5: Convergence plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = {'0.5': 'C0', '0.9': 'C1', '1.0': 'C2'}
for sigma in sigma_vals:
    sk = str(sigma)
    ax.loglog(Nx_list, results[sk]['L2'], 'o-', color=colors[sk], 
              label=f'$\\sigma$={sigma}')
Nx_ref = np.array(Nx_list)
ax.loglog(Nx_ref, 1e-2/Nx_ref**2, 'k--', alpha=0.5, label=r'$\mathcal{O}(\Delta x^2)$')
ax.set_xlabel('$N_x$', fontsize=14)
ax.set_ylabel('$L_2$ error', fontsize=14)
ax.set_title(f'Grid convergence ($T={T_conv}$, five periods)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig5_convergence.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Fig 5 saved")

# Fig 6: Error vs sigma with Nx=100,200,400
fig2, ax2 = plt.subplots(figsize=(8, 5))
sigma_range = np.linspace(0.1, 1.0, 19)
for Nx_fixed, ls, label in [(100, 'o-', '$N_x=100$'), (200, 's--', '$N_x=200$'), (400, 'd-.', '$N_x=400$')]:
    L2_sigma = []
    for sigma in sigma_range:
        L2, _, _, _, _ = solve_wave(L, c, T=T_conv, Nx=Nx_fixed, sigma=sigma)
        L2_sigma.append(L2)
    ax2.semilogy(sigma_range, L2_sigma, ls, label=label)
ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.7, label='$\\sigma=1$ (exact)')
ax2.set_xlabel('Courant number $\\sigma$', fontsize=14)
ax2.set_ylabel('$L_2$ error', fontsize=14)
ax2.set_title(f'Error vs Courant number ($T={T_conv}$)', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig6_error_vs_sigma.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Fig 6 saved (3 curves: Nx=100,200,400)")

# Print summary
print("\n===== SUMMARY =====")
for sigma in sigma_vals:
    sk = str(sigma)
    rates = results[sk]['rate_L2']
    print(f"sigma={sigma:.1f}: rates={rates}, mean={np.mean(rates):.2f}")
