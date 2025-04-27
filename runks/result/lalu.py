import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Данные
n = np.array([2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
t_median = np.array([0.271, 2.275, 7.8, 18.063, 31.684, 63.64, 97.56, 150.827])

# Степенная функция: T(n) = b + a * n^3
def power_law(n, a, b):
    return b + a * n**3

# Степенная функция в целом: T(n) = b + a * n^с
def power_law2(n, a, b, с):
    return b + a * n**с

# Аппроксимация
params, _ = curve_fit(power_law, n, t_median)
a, b = params
# Аппроксимация в целом
params2, _ = curve_fit(power_law2, n, t_median)
a2, b2, c2 = params2

# Предсказанные значения
n_fit = np.linspace(2500, 20000, 100)
t_fit = power_law(n_fit, a, b)
t_fit2 = power_law2(n_fit, a2, b2, c2)

# График
plt.figure(figsize=(14, 8))
plt.scatter(n, t_median, color='red', label='Медианное время (измерение)', zorder=5)

# Добавляем подписи к точкам
for x, y in zip(n, t_median):
    plt.text(
        x, y + 200,  # Смещение по y для читаемости
        f'n={x}\nt={y:.1f} с', 
        fontsize=8, 
        ha='center', 
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

plt.plot(n_fit, t_fit, '--', label=f'Аппроксимация: $T(n) = {b:.2e} + {a:.2e} \cdot n^3$')
plt.plot(n_fit, t_fit2, '--', label=f'Общая аппроксимация: $T(n) = {b2:.2e} + {a2:.2e} \cdot n^{{{c2:.2f}}}$')

plt.xlabel('Размер матрицы ($n$)', fontsize=12)
plt.ylabel('Время (с)', fontsize=12)
plt.title('Зависимость времени обращения матрицы от её размера (LU-обращение)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
