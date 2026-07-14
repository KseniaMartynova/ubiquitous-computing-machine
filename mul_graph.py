import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import glob
# ------------------------------------------------------------
# ОПРЕДЕЛЯЕМ БАЗОВУЮ ДИРЕКТОРИЮ 
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# ПАПКИ С РЕЗУЛЬТАТАМИ 
# ------------------------------------------------------------
LAPACK_DIR = os.path.join(BASE_DIR, "runks", "build", "lapack", "lu", "results")
MKL_DIR    = os.path.join(BASE_DIR, "runks", "build", "mkl", "lu", "results")
NUMPY_DIR  = os.path.join(BASE_DIR, "runks", "build", "numpy", "lu", "results")

PREFIXES = {
    'LAPACK': 'lapack',
    'MKL':    'mkl',
    'NumPy':  'num'      
}

# ------------------------------------------------------------
# ЧТЕНИЕ ДАННЫХ ИЗ ФАЙЛОВ
# ------------------------------------------------------------
def read_times(directory, prefix):
    """
    Считывает размеры матриц и времена из файлов вида:
        <prefix>_mul_size_<N>.txt
    Возвращает отсортированные массивы (n, time).
    """
    pattern = os.path.join(directory, f"{prefix}_mul_size_*.txt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Не найдены файлы по шаблону: {pattern}")

    sizes = []
    times = []
    for fpath in files:
        # Извлекаем размер из имени файла (между "size_" и ".txt")
        basename = os.path.basename(fpath)
        size_str = basename.split("_size_")[1].replace(".txt", "")
        n_val = int(size_str)
        with open(fpath, "r") as f:
            t_val = float(f.read().strip())
        sizes.append(n_val)
        times.append(t_val)

    # Сортируем по размеру матрицы
    order = np.argsort(sizes)
    n_sorted = np.array(sizes)[order]
    t_sorted = np.array(times)[order]
    return n_sorted, t_sorted

# Читаем данные для трёх библиотек
n_lapack, t_lapack = read_times(LAPACK_DIR, "lapack")
n_mkl,    t_mkl    = read_times(MKL_DIR,    "mkl")
n_numpy,  t_numpy  = read_times(NUMPY_DIR,  "numpy")


print("Загруженные данные (LAPACK):", list(zip(n_lapack, t_lapack)))
print("Загруженные данные (MKL):",    list(zip(n_mkl, t_mkl)))
print("Загруженные данные (NumPy):",  list(zip(n_numpy, t_numpy)))

# ------------------------------------------------------------
# АППРОКСИМАЦИЯ 
# ------------------------------------------------------------
# Для LAPACK и NumPy используем все точки
log_n_lapack = np.log(n_lapack)
log_t_lapack = np.log(t_lapack)

log_n_numpy = np.log(n_numpy)
log_t_numpy = np.log(t_numpy)


log_n_mkl = np.log(n_mkl)
log_t_mkl = np.log(t_mkl)

# Линейная модель в логарифмическом пространстве: log(T) = log(a) + c*log(n)
def linear_log_model(log_n, log_a, c):
    return log_a + c * log_n

params_lapack, _ = curve_fit(linear_log_model, log_n_lapack, log_t_lapack)
params_mkl,    _ = curve_fit(linear_log_model, log_n_mkl,    log_t_mkl)
params_numpy,  _ = curve_fit(linear_log_model, log_n_numpy,  log_t_numpy)

a_lapack, c_lapack = np.exp(params_lapack[0]), params_lapack[1]
a_mkl,    c_mkl    = np.exp(params_mkl[0]),    params_mkl[1]
a_numpy,  c_numpy  = np.exp(params_numpy[0]),  params_numpy[1]

print(f"Показатель степени (LAPACK): {c_lapack:.3f}")
print(f"Показатель степени (MKL):    {c_mkl:.3f}")
print(f"Показатель степени (NumPy):  {c_numpy:.3f}")

# ------------------------------------------------------------
# ПОСТРОЕНИЕ ГРАФИКА
# ------------------------------------------------------------
plt.figure(figsize=(14, 8))
plt.xscale('log')
plt.yscale('log')

# Настройка делений осей
plt.xticks(
    ticks=[2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000],
    labels=['2.5k', '5k', '7.5k', '10k', '12.5k', '15k', '17.5k', '20k'],
    fontsize=10
)

plt.yticks(
    ticks=[0.01, 0.025, 0.05, 0.09, 0.16, 0.3, 0.5, 1, 1.7, 3, 5, 10, 20, 40, 70, 100],
    labels=['0.01', '0.025', '0.05', '0.09', '0.16', '0.3', '0.5', '1', '1.7',
            '3', '5', '10', '20', '40', '70', '100'],
    fontsize=10
)

# Точки измерений
plt.scatter(n_lapack, t_lapack, color='green', label='LAPACK (измерения)', zorder=5)
plt.scatter(n_mkl,    t_mkl,    color='purple', marker='s', label='MKL (измерения)', zorder=5)
plt.scatter(n_numpy,  t_numpy,  color='brown', marker='^', label='NumPy (измерения)', zorder=5)

# Аппроксимирующие кривые
n_fit = np.linspace(2000, 21000, 500)
plt.plot(n_fit, a_lapack * n_fit**c_lapack, '--', color='green', alpha=0.7,
         label=f'LAPACK: $T(n) = n^{{{c_lapack:.2f}}}$')
plt.plot(n_fit, a_mkl    * n_fit**c_mkl,    '--', color='purple', alpha=0.7,
         label=f'MKL: $T(n) = n^{{{c_mkl:.2f}}}$')
plt.plot(n_fit, a_numpy  * n_fit**c_numpy,  '--', color='brown', alpha=0.7,
         label=f'NumPy: $T(n) = n^{{{c_numpy:.2f}}}$')

# Подписи значений времени около точек
for x, y1, y2, y3 in zip(n_lapack, t_lapack, t_mkl, t_numpy):
    plt.text(x, y1*1.4, f'{y1:.1f}s', ha='center', va='bottom',
             color='green', fontsize=8, fontweight='bold')
    plt.text(x, y2*0.6, f'{y2:.1f}s', ha='center', va='top',
             color='purple', fontsize=8, fontweight='bold')
    plt.text(x, y3*1.4, f'{y3:.1f}s', ha='center', va='bottom',
             color='brown', fontsize=8, fontweight='bold')

plt.xlabel('Размер матрицы ($n$)', fontsize=12, labelpad=10)
plt.ylabel('Время (с)', fontsize=12, labelpad=10)
plt.title('Аппроксимация времени умножения (линейная регрессия в log-log)',
          fontsize=14, pad=15)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5, which='both')

plt.tight_layout()
plt.show()
