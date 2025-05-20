import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Данные
n = np.array([2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
t_lapack = np.array([0.114659, 0.605983, 1.659070, 3.859710, 7.170840, 13.134900, 18.529300, 31.653600])
t_mkl = np.array([0.067106, 0.393082, 1.19357, 2.91897, 6.10533, 9.94007, 15.1867, 22.4722])
t_numpy = np.array([0.266776, 1.593896, 5.161883, 11.854884, 23.477536, 39.750357, 60.445645, 92.938226])

# Линейная модель в логарифмическом пространстве: log(T) = log(a) + c*log(n)
def linear_log_model(log_n, log_a, c):
    return log_a + c * log_n

# Преобразование данных
log_n = np.log(n)
log_t_lapack = np.log(t_lapack)
log_t_mkl = np.log(t_mkl)
log_t_numpy = np.log(t_numpy)
# Аппроксимация методом наименьших квадратов
params_lapack, _ = curve_fit(linear_log_model, log_n, log_t_lapack)
params_mkl, _ = curve_fit(linear_log_model, log_n, log_t_mkl)
params_numpy, _ = curve_fit(linear_log_model, log_n, log_t_numpy)

# Коэффициенты моделей
a_lapack, c_lapack = np.exp(params_lapack[0]), params_lapack[1]
a_mkl, c_mkl = np.exp(params_mkl[0]), params_mkl[1]
a_numpy, c_numpy = np.exp(params_numpy[0]), params_numpy[1]

# Построение графика
plt.figure(figsize=(14, 8))
plt.xscale('log')
plt.yscale('log')

# Настройка делений и подписей осей
plt.xticks(
    ticks=[2500, 5000,7500, 10000, 12500,15000,17500, 20000], 
    labels=['2.5k', '5k', '7.5k', '10k', '12.5k','15k','17.5k', '20k'],
    fontsize=10
)
plt.yticks(
    ticks=[0, 1, 5, 10,50, 80, 100], 
    labels=['0', '1','5', '10','50','80', '100'],
    fontsize=10
)

# Точки данных
plt.scatter(n, t_lapack, color='green', label='LAPACK', zorder=5)
plt.scatter(n, t_mkl, color='purple', marker='s', label='MKL', zorder=5)
plt.scatter(n, t_numpy, color='brown', marker='^', label='NumPy', zorder=5)
# Аппроксимационные кривые
n_fit = np.linspace(2000, 21000, 500)
plt.plot(n_fit, a_lapack * n_fit**c_lapack, '--', color='green', alpha=0.7, 
         label=f'LAPACK: $T(n) = n^{{{c_lapack:.2f}}}$')
plt.plot(n_fit, a_mkl * n_fit**c_mkl, '--', color='purple', alpha=0.7,
         label=f'MKL: $T(n) = n^{{{c_mkl:.2f}}}$')
plt.plot(n_fit, a_numpy * n_fit**c_numpy, '--', color='brown', alpha=0.7,
         label=f'NumPy: $T(n) = n^{{{c_numpy:.2f}}}$')

# Подписи точек (исправлены цвета)
for x, y1, y2, y3 in zip(n, t_lapack, t_mkl, t_numpy):
    plt.text(x, y1*1.5, f'{y1:.1f}s', ha='center', va='bottom', 
             color='green', fontsize=8, fontweight='bold')
    plt.text(x, y2*0.7, f'{y2:.1f}s', ha='center', va='top', 
             color='purple', fontsize=8, fontweight='bold')
    plt.text(x, y3*1.5, f'{y3:.1f}s', ha='center', va='bottom', 
             color='brown', fontsize=8, fontweight='bold')

# Оформление
plt.xlabel('Размер матрицы ($n$)', fontsize=12, labelpad=10)
plt.ylabel('Время (s)', fontsize=12, labelpad=10)
plt.title('Аппроксимация времени обращения Холецкого (линейная регрессия в log-log)', 
          fontsize=14, pad=15)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5, which='both')

plt.tight_layout()
plt.show()
