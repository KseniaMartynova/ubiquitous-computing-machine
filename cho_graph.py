import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

n = np.array([2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
t_lapack = np.array([0.2369593, 0.9693086, 2.669331, 5.70002,
                     10.38484, 16.96723, 25.79507, 34.22719])
t_mkl = np.array([0.10471931, 0.550439, 1.541744, 3.481571, 6.726509,
                  11.36498, 17.95102, 25.08364])
t_numpy = np.array([0.5686822, 2.1517803, 5.8037797, 12.7761449,
                    21.0688445, 35.538212, 53.666742, 82.8009671])

# ------------------------------
# Подготовка данных для регрессии MKL (исключаем первое значение)
# ------------------------------
mkl_indices = np.arange(0, len(n))   # используем точки со 2-й по последнюю
n_mkl = n[mkl_indices]
t_mkl_fit = t_mkl[mkl_indices]       # времена для регрессии

# Линейная модель в логарифмическом пространстве: log(T) = log(a) + c*log(n)
def linear_log_model(log_n, log_a, c):
    return log_a + c * log_n

# Логарифмируем все данные 
log_n_all = np.log(n)
log_t_lapack = np.log(t_lapack)
log_t_numpy = np.log(t_numpy)

# Для MKL используем только отобранные индексы
log_n_mkl = np.log(n_mkl)
log_t_mkl = np.log(t_mkl_fit)

# Аппроксимация методом наименьших квадратов
params_lapack, _ = curve_fit(linear_log_model, log_n_all, log_t_lapack)
params_mkl, _ = curve_fit(linear_log_model, log_n_mkl, log_t_mkl)
params_numpy, _ = curve_fit(linear_log_model, log_n_all, log_t_numpy)

# Коэффициенты моделей (a и c)
a_lapack, c_lapack = np.exp(params_lapack[0]), params_lapack[1]
a_mkl, c_mkl = np.exp(params_mkl[0]), params_mkl[1]
a_numpy, c_numpy = np.exp(params_numpy[0]), params_numpy[1]

print(f"Показатель степени (LAPACK): {c_lapack:.3f}")
print(f"Показатель степени (MKL):    {c_mkl:.3f}")
print(f"Показатель степени (NumPy):  {c_numpy:.3f}")

plt.figure(figsize=(14, 8))
plt.xscale('log')
plt.yscale('log')

# Настройка делений и подписей осей (убираем 0 из Y, заменяем на 0.1)
plt.xticks(
    ticks=[2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000],
    labels=['2.5k', '5k', '7.5k', '10k', '12.5k', '15k', '17.5k', '20k'],
    fontsize=10
)

plt.yticks(
    ticks=[0.01, 0.025,0.05,0.09,0.16,0.3,0.5,1,1.7, 3, 5, 10, 20, 40, 70,100],
    labels=['0.01', '0.025','0.05','0.09','0.16','0.3','0.5','1','1.7','3', '5', '10', '20','40', '70','100'],
    fontsize=10
)

# Точки данных
plt.scatter(n, t_lapack, color='green', label='LAPACK (измерения)', zorder=5)
plt.scatter(n, t_mkl, color='purple', marker='s', label='MKL (измерения)', zorder=5)
plt.scatter(n, t_numpy, color='brown', marker='^', label='NumPy (измерения)', zorder=5)

# Аппроксимационные кривые (для MKL строим только по точкам, использованным в регрессии, но отображаем на всём диапазоне)
n_fit = np.linspace(2000, 21000, 500)
plt.plot(n_fit, a_lapack * n_fit**c_lapack, '--', color='green', alpha=0.7,
         label=f'LAPACK: $T(n) = n^{{{c_lapack:.2f}}}$')
plt.plot(n_fit, a_mkl * n_fit**c_mkl, '--', color='purple', alpha=0.7,
         label=f'MKL: $T(n) = n^{{{c_mkl:.2f}}}$')
plt.plot(n_fit, a_numpy * n_fit**c_numpy, '--', color='brown', alpha=0.7,
         label=f'NumPy: $T(n) = n^{{{c_numpy:.2f}}}$')

# Подписи точек 
for x, y1, y2, y3 in zip(n, t_lapack, t_mkl, t_numpy):
    plt.text(x, y1*1.4, f'{y1:.1f}s', ha='center', va='bottom',
             color='green', fontsize=8, fontweight='bold')
    plt.text(x, y2*0.6, f'{y2:.1f}s', ha='center', va='top',
             color='purple', fontsize=8, fontweight='bold')
    plt.text(x, y3*1.4, f'{y3:.1f}s', ha='center', va='bottom',
             color='brown', fontsize=8, fontweight='bold')

plt.xlabel('Размер матрицы ($n$)', fontsize=12, labelpad=10)
plt.ylabel('Время (с)', fontsize=12, labelpad=10)
plt.title('Аппроксимация времени обращения Холецкого (линейная регрессия в log-log)',
          fontsize=14, pad=15)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5, which='both')

plt.tight_layout()
plt.show()
