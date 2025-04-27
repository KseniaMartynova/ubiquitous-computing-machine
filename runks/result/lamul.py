import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Данные
n = np.array([2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
t_median = np.array([0.061, 0.508, 1.799, 4.249, 16.492, 26.126, 36.362, 56.285])

# 1. Кубическая функция: T(n) = b + a * n^3
def cubic_law(n, a, b):
    return b + a * n**3

# 2. Общая степенная функция: T(n) = b + a * n^c
def general_power_law(n, a, b, c):
    return b + a * n**c

# Аппроксимация кубической зависимостью
params_cubic, _ = curve_fit(cubic_law, n, t_median, p0=[1e-11, 0], maxfev=5000)
a_cubic, b_cubic = params_cubic

# Аппроксимация общей степенной зависимостью
params_gen, _ = curve_fit(general_power_law, n, t_median, p0=[1e-11, 0, 3], maxfev=5000)
a_gen, b_gen, c_gen = params_gen

# Предсказанные значения
n_fit = np.linspace(2500, 20000, 100)
t_fit_cubic = cubic_law(n_fit, a_cubic, b_cubic)
t_fit_gen = general_power_law(n_fit, a_gen, b_gen, c_gen)

# График
plt.figure(figsize=(14, 8))
plt.scatter(n, t_median, color='red', label='Медианное время (измерение)', zorder=5)

# Подписи к точкам
for x, y in zip(n, t_median):
    plt.text(x, y + 1, f'n={x}\nt={y:.1f} с', fontsize=8, ha='center', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.plot(n_fit, t_fit_cubic, '--', 
         label=f'Кубическая: $T(n) = {b_cubic:.2e} + {a_cubic:.2e} \cdot n^3$')
plt.plot(n_fit, t_fit_gen, '--', 
         label=f'Общая степенная: $T(n) = {b_gen:.2e} + {a_gen:.2e} \cdot n^{{{c_gen:.2f}}}$')

plt.xlabel('Размер матрицы ($n$)', fontsize=12)
plt.ylabel('Время (с)', fontsize=12)
plt.title('Умножение lapack', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
