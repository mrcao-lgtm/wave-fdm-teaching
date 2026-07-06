#!/usr/bin/env python3
"""
Grid convergence study for wave equation FDM.
Tests L2-norm convergence rate as Nx increases.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json

# ----- solver (same as wave_fdm.py) -----
def solve_wave(L=1.0, c=1.0, T=2.0, Nx=100, sigma=0.9):
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

# ----- grid convergence -----
output_dir = r"D:\飞书协作群1资料\14_波动方程FDM教学\02_研究执行\data"
os.makedirs(output_dir, exist_ok=True)
fig_dir = r"D:\飞书协作群1资料\14_波动方程FDM教学\02_研究执行\figures"

L, c = 1.0, 1.0
Nx_list = [20, 40, 80, 160, 320, 640, 1280]
sigma_vals = [0.5, 0.9, 1.0]

results = {}
fig, ax = plt.subplots(figsize=(8, 6))
colors = {'0.5': 'C0', '0.9': 'C1', '1.0': 'C2'}

for sigma in sigma_vals:
    L2_vals = []
    Linf_vals = []
    key = str(sigma)
    for Nx in Nx_list:
        L2, Linf, x, u_num, u_ex = solve_wave(L, c, T=2.0, Nx=Nx, sigma=sigma)
        L2_vals.append(L2)
        Linf_vals.append(Linf)
        print(f"  sigma={sigma:.1f}, Nx={Nx:4d}: L2={L2:.3e}, Linf={Linf:.3e}")
    results[key] = {'Nx': Nx_list, 'L2': L2_vals, 'Linf': Linf_vals}
    # convergence rate
    rates = [np.log(L2_vals[i]/L2_vals[i+1])/np.log(2) for i in range(len(L2_vals)-1)]
    results[key]['rate_L2'] = rates
    print(f"  sigma={sigma:.1f} L2 convergence rates: {[f'{r:.2f}' for r in rates]}")
    ax.loglog(Nx_list, L2_vals, 'o-', color=colors[key], label=f'$\\sigma$={sigma}')
    
# Second-order reference line
Nx_ref = np.array(Nx_list)
ax.loglog(Nx_ref, 0.1/Nx_ref**2, 'k--', alpha=0.5, label=r'$\mathcal{O}(\Delta x^2)$')
ax.set_xlabel('$N_x$')
ax.set_ylabel('$L_2$ error')
ax.set_title('Grid convergence of wave equation FDM')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig5_convergence.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Fig 5 saved")

# ----- also do a sigma-scan at fixed Nx to show error vs sigma -----
Nx_fixed = 100
sigma_range = np.linspace(0.1, 1.0, 19)
L2_sigma = []
for sigma in sigma_range:
    L2, Linf, _, _, _ = solve_wave(L, c, T=2.0, Nx=Nx_fixed, sigma=sigma)
    L2_sigma.append(L2)
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.semilogy(sigma_range, L2_sigma, 'o-', color='C3')
ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.7, label='$\\sigma=1$ (exact)')
ax2.set_xlabel('Courant number $\\sigma$')
ax2.set_ylabel('$L_2$ error')
ax2.set_title(f'Error vs Courant number ($N_x={Nx_fixed}$)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig6_error_vs_sigma.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Fig 6 saved")

# save raw data
data_path = os.path.join(output_dir, 'convergence_data.json')
with open(data_path, 'w') as f:
    json.dump({k: {sk: sv if isinstance(sv, list) else [float(v) for v in sv] 
                   for sk, sv in results.items()}, 
              'sigma_scan': {'sigma': sigma_range.tolist(), 'L2': L2_sigma}},
             f, indent=2)
print(f"✅ Data saved to {data_path}")

print("\n===== SUMMARY =====")
for sigma in sigma_vals:
    key = str(sigma)
    rates = results[key]['rate_L2']
    print(f"sigma={sigma:.1f}: avg rate = {np.mean(rates):.2f}")
