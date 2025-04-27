import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
n = np.array([2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
t_median = np.array([0.114659, 0.605983, 1.659070, 3.859710, 7.170840, 13.134900, 18.529300, 31.653600])

# Power law function: T(n) = b + a * n^3
def power_law(n, a, b):
    return b + a * n**3

# General power law function: T(n) = b + a * n^c
def power_law2(n, a, b, c):
    return b + a * n**c

# Fit for cubic law
params, _ = curve_fit(power_law, n, t_median)
a, b = params

# Fit for general power law with increased max iterations and initial guess
params2, _ = curve_fit(power_law2, n, t_median, 
                      p0=[1e-10, 0, 3],  # Initial guess near expected values
                      maxfev=5000)       # Increased maximum function evaluations
a2, b2, c2 = params2

# Predicted values
n_fit = np.linspace(2500, 20000, 100)
t_fit = power_law(n_fit, a, b)
t_fit2 = power_law2(n_fit, a2, b2, c2)

# Plot
plt.figure(figsize=(14, 8))
plt.scatter(n, t_median, color='red', label='Медианное время', zorder=5)

# Add point labels
for x, y in zip(n, t_median):
    plt.text(
        x, y + 1,  # y-offset adjusted
        f'n={x}\nt={y:.1f} s', 
        fontsize=8, 
        ha='center', 
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

plt.plot(n_fit, t_fit, '--', label=f'Аппроксимация: $T(n) = {b:.2e} + {a:.2e} \cdot n^3$')
plt.plot(n_fit, t_fit2, '--', label=f'Общая аппроксимация: $T(n) = {b2:.2e} + {a2:.2e} \cdot n^{{{c2:.2f}}}$')

plt.xlabel('Размер матрицы ($n$)', fontsize=12)
plt.ylabel('Время (s)', fontsize=12)
plt.title('Время обращения (SVD)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
