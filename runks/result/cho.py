import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Исходные данные (первый набор измерений)
n = np.array([2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
t_median1 = np.array([0.114659, 0.605983, 1.659070, 3.859710, 7.170840, 13.134900, 18.529300, 31.653600])#lapach

# Второй набор данных (например, другие измерения для того же метода)
t_median2 = np.array([0.067106, 0.393082, 1.19357, 2.91897, 6.10533, 9.94007, 15.1867, 22.4722])  # mkl
t_median3 = np.array([0.266776,1.593896, 5.161883, 11.854884, 23.477536, 39.750357, 60.445645, 92.938226])  # numpy
# Степенная функция: T(n) = b + a * n^3
def power_law(n, a, b):
    return b + a * n**3

# Общая степенная функция: T(n) = b + a * n^c
def power_law2(n, a, b, c):
    return b + a * n**c

# Аппроксимация для первого набора данных
params1, _ = curve_fit(power_law, n, t_median1)
a1, b1 = params1
params1_gen, _ = curve_fit(power_law2, n, t_median1, p0=[1e-10, 0, 3], maxfev=5000)
a1g, b1g, c1g = params1_gen

# Аппроксимация для второго набора данных
params2, _ = curve_fit(power_law, n, t_median2)
a2, b2 = params2
params2_gen, _ = curve_fit(power_law2, n, t_median2, p0=[1e-10, 0, 3], maxfev=5000)
a2g, b2g, c2g = params2_gen

params3, _ = curve_fit(power_law, n, t_median3)
a3, b3 = params3
params3_gen, _ = curve_fit(power_law2, n, t_median3, p0=[1e-10, 0, 3], maxfev=5000)
a3g, b3g, c3g = params3_gen

# Предсказанные значения
n_fit = np.linspace(2500, 20000, 100)
t_fit1 = power_law(n_fit, a1, b1)
t_fit1g = power_law2(n_fit, a1g, b1g, c1g)
t_fit2 = power_law(n_fit, a2, b2)
t_fit2g = power_law2(n_fit, a2g, b2g, c2g)
t_fit3 = power_law(n_fit, a3, b3)
t_fit3g = power_law2(n_fit, a3g, b3g, c3g)

# Построение графика
plt.figure(figsize=(14, 8))

# Первый набор данных
plt.scatter(n, t_median1, color='green', label='Измерение lapack (медиана)', zorder=5)
plt.plot(n_fit, t_fit1, '--', color='red', alpha=0.5, label=f'Аппроксимация 1: $T(n) = {b1:.2e} + {a1:.2e}n^3$')
plt.plot(n_fit, t_fit1g, '--', color='green', alpha=0.5, label=f'Общая аппроксимация 1: $T(n) = {b1g:.2e} + {a1g:.2e}n^{{{c1g:.2f}}}$')

# Второй набор данных
plt.scatter(n, t_median2, color='blue', marker='s', label='Измерение mkl (медиана)', zorder=5)
#plt.plot(n_fit, t_fit2, '--', color='blue', alpha=0.5, label=f'Аппроксимация 2: $T(n) = {b2:.2e} + {a2:.2e}n^3$')
plt.plot(n_fit, t_fit2g, '--', color='blue', alpha=0.5, label=f'Общая аппроксимация 2: $T(n) = {b2g:.2e} + {a2g:.2e}n^{{{c2g:.2f}}}$')

# Второй набор данных
plt.scatter(n, t_median3, color='orange', marker='s', label='Измерение numpy (медиана)', zorder=5)
#plt.plot(n_fit, t_fit2, '--', color='blue', alpha=0.5, label=f'Аппроксимация 2: $T(n) = {b2:.2e} + {a2:.2e}n^3$')
plt.plot(n_fit, t_fit3g, '--', color='orange', alpha=0.5, label=f'Общая аппроксимация 3: $T(n) = {b3g:.2e} + {a3g:.2e}n^{{{c3g:.2f}}}$')
# Подписи точек
for x, y1, y2 in zip(n, t_median1, t_median2):
    plt.text(x, y1 + 1, f'n={x}\n{y1:.2f} с', fontsize=8, ha='center', color='red')
    plt.text(x, y2 + 1, f'n={x}\n{y2:.2f} с', fontsize=8, ha='center', color='blue', va='top')

plt.xlabel('Размер матрицы ($n$)', fontsize=12)
plt.ylabel('Время (с)', fontsize=12)
plt.title('Сравнение измерений времени обращения (Холецкий)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
