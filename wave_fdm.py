#!/usr/bin/env python3
"""
Teaching Wave Propagation with the Finite Difference Method
Numerical experiments for the 1D wave equation

Experiments:
1. Standing wave verification
2. Traveling wave packet
3. Boundary reflections (fixed/free)
4. Numerical dispersion vs CFL number
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

# ====== Wave equation solver ======
def solve_wave(L=1.0, c=1.0, T=5.0, Nx=100, sigma=0.9,
               init_type='standing', x0=0.3, delta=0.05):
    dx = L / Nx
    dt = sigma * dx / c
    Nt = int(T / dt)
    x = np.linspace(0, L, Nx+1)
    
    u_prev = np.zeros(Nx+1)
    u_curr = np.zeros(Nx+1)
    
    if init_type == 'standing':
        u_curr = np.sin(np.pi * x / L)
        u_prev = u_curr.copy()
    elif init_type == 'traveling':
        u_curr = np.exp(-(x - x0)**2 / (2*delta**2))
        v0 = -c * np.gradient(u_curr, dx)
        u_prev = u_curr - dt * v0 + 0.5 * sigma**2 * (
            np.roll(u_curr, -1) - 2*u_curr + np.roll(u_curr, 1))
    elif init_type == 'gaussian_pulse':
        u_curr = np.exp(-(x - x0)**2 / (2*delta**2))
        u_prev = u_curr.copy()
    
    u_curr[0] = u_curr[-1] = 0
    u_prev[0] = u_prev[-1] = 0
    
    save_interval = max(1, Nt // 20)
    saved = [(0, u_curr.copy())]
    sigma2 = sigma**2
    
    for n in range(Nt):
        u_next = np.zeros(Nx+1)
        u_next[1:-1] = (2*(1-sigma2) * u_curr[1:-1] +
                       sigma2 * (u_curr[2:] + u_curr[:-2]) -
                       u_prev[1:-1])
        u_next[0] = u_next[-1] = 0
        u_prev = u_curr.copy()
        u_curr = u_next.copy()
        if (n+1) % save_interval == 0:
            saved.append(((n+1)*dt, u_curr.copy()))
    
    t_saved = [s[0] for s in saved]
    u_saved = [s[1] for s in saved]
    return u_saved, x, t_saved, dt, Nt


def solve_wave_free(L=1.0, c=1.0, T=3.0, Nx=100, sigma=0.9, x0=0.3, delta=0.05):
    dx = L / Nx
    dt = sigma * dx / c
    Nt = int(T / dt)
    x = np.linspace(0, L, Nx+1)
    u_prev = np.zeros(Nx+1)
    u_curr = np.exp(-(x - x0)**2 / (2*delta**2))
    u_prev = u_curr.copy()
    sigma2 = sigma**2
    save_interval = max(1, Nt // 20)
    saved = [(0, u_curr.copy())]
    for n in range(Nt):
        u_next = np.zeros(Nx+1)
        u_next[1:-1] = (2*(1-sigma2) * u_curr[1:-1] +
                       sigma2 * (u_curr[2:] + u_curr[:-2]) -
                       u_prev[1:-1])
        u_next[0] = u_next[1]
        u_next[-1] = u_next[-2]
        u_prev = u_curr.copy()
        u_curr = u_next.copy()
        if (n+1) % save_interval == 0:
            saved.append(((n+1)*dt, u_curr.copy()))
    return [s[1] for s in saved], x, [s[0] for s in saved], dt, Nt


# ====== Generate all figures ======
fig_dir = r"D:\飞书协作群1资料\14_波动方程FDM教学\02_研究执行\figures"
os.makedirs(fig_dir, exist_ok=True)
L, c = 1.0, 1.0

# Fig 1: Standing wave
print("Fig 1: Standing wave...")
us, x, ts, dt, Nt = solve_wave(L, c, T=3.0, Nx=100, sigma=0.9, init_type='standing')
fig, ax = plt.subplots(figsize=(8, 5))
t_plot = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_plot)))
for i, t in enumerate(t_plot):
    u_exact = np.sin(np.pi*x/L) * np.cos(np.pi*c*t/L)
    idx = min(range(len(ts)), key=lambda j: abs(ts[j]-t))
    label = f'$t={t:.1f}$' if i in [0, 2, 4, 6] else None
    ax.plot(x, us[idx], 'o', color=colors[i], markersize=3, alpha=0.6)
    ax.plot(x, u_exact, '-', color=colors[i], linewidth=1.5, alpha=0.4, label=label)
ax.set_xlabel('$x$ (m)'); ax.set_ylabel('$u(x,t)$')
ax.set_title('Standing wave: numerical (points) vs analytical (lines), $\\sigma=0.9$')
ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig1_standing_wave.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

# Fig 2: Traveling wave packet
print("Fig 2: Traveling wave packet...")
us, x, ts, dt, Nt = solve_wave(L, c, T=2.0, Nx=100, sigma=0.9, init_type='traveling')
fig, ax = plt.subplots(figsize=(8, 5))
plot_indices = [0, 5, 10, 15, 19]
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(plot_indices)))
for i, idx in enumerate(plot_indices):
    if idx < len(us):
        ax.plot(x, us[idx], color=colors[i], linewidth=2, label=f'$t={ts[idx]:.2f}$')
ax.set_xlabel('$x$ (m)'); ax.set_ylabel('$u(x,t)$')
ax.set_title('Gaussian wave packet propagating to the right ($\\sigma=0.9$)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig2_traveling_wave.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

# Fig 3: Boundary reflections
print("Fig 3: Boundary reflections...")
us_fixed, x, ts, _, _ = solve_wave(L, c, T=3.0, Nx=100, sigma=0.9, init_type='gaussian_pulse', x0=0.3, delta=0.05)
us_free, _, _, _, _ = solve_wave_free(L, c, T=3.0, Nx=100, sigma=0.9, x0=0.3, delta=0.05)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_idx = [0, 5, 10, 15, 19]
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(plot_idx)))
for i, idx in enumerate(plot_idx):
    if idx < len(us_fixed):
        ax1.plot(x, us_fixed[idx], color=colors[i], linewidth=1.5, label=f'$t={ts[idx]:.2f}$')
ax1.set_xlabel('$x$ (m)'); ax1.set_ylabel('$u(x,t)$')
ax1.set_title('(a) Fixed end reflection'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
for i, idx in enumerate(plot_idx):
    if idx < len(us_free):
        ax2.plot(x, us_free[idx], color=colors[i], linewidth=1.5, label=f'$t={ts[idx]:.2f}$')
ax2.set_xlabel('$x$ (m)'); ax2.set_ylabel('$u(x,t)$')
ax2.set_title('(b) Free end reflection'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig3_reflection.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

# Fig 4: Numerical dispersion
print("Fig 4: Numerical dispersion...")
sigmas_test = [0.3, 0.6, 0.9, 1.0]
colors_disp = plt.cm.coolwarm(np.linspace(0.2, 0.9, len(sigmas_test)))
fig, ax = plt.subplots(figsize=(8, 5))
x0, delta = 0.3, 0.05
for sigma_val, color in zip(sigmas_test, colors_disp):
    us_disp, x_disp, ts_disp, _, _ = solve_wave(
        L, c, T=2.0, Nx=100, sigma=sigma_val, init_type='traveling', x0=x0, delta=delta)
    if len(us_disp) > 1:
        ax.plot(x_disp, us_disp[-1], color=color, linewidth=2, label=f'$\\sigma={sigma_val}$')
u_init = np.exp(-(x-x0)**2/(2*delta**2))
ax.plot(x, u_init, 'k--', alpha=0.5, linewidth=1.5, label='Initial')
ax.set_xlabel('$x$ (m)'); ax.set_ylabel('$u(x,T)$')
ax.set_title('Numerical dispersion at different Courant numbers')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig4_dispersion.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print("  ✅")

print(f"\nAll figures done! Saved to {fig_dir}")
for f in sorted(os.listdir(fig_dir)):
    fp = os.path.join(fig_dir, f)
    print(f"  {f:40s} {os.path.getsize(fp)/1024:6.1f}KB")
