import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import glob
import re

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

# Префиксы имён файлов: lapack_lu_size_<N>.txt, mkl_lu_size_<N>.txt, num_lu_size_<N>.txt
PREFIXES = {
    'LAPACK': 'lapack',
    'MKL':    'mkl',
    'NumPy':  'num'      # или 'numpy' – проверьте, как у вас реально называются файлы
}

# ------------------------------------------------------------
# ЧТЕНИЕ ДАННЫХ ИЗ ФАЙЛОВ 
# ------------------------------------------------------------
def read_times(directory, prefix):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Директория не существует: {directory}")
    
    pattern = os.path.join(directory, f"{prefix}_lu_size_*.txt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"Не найдены файлы по шаблону: {pattern}\n"
            f"Проверьте, что в папке {directory} есть файлы типа {prefix}_lu_size_2500.txt"
        )
    
    sizes = []
    avg_times = []
    
    for fpath in files:
        basename = os.path.basename(fpath)
        size_str = basename.split("_size_")[1].replace(".txt", "")
        n_val = int(size_str)
        
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            # Ищем все числа вида ": число s"
            values = re.findall(r":\s*([\d.]+)\s*s", content)
            if not values:
                try:
                    single_val = float(content.strip())
                    values = [str(single_val)]
                except ValueError:
                    raise ValueError(f"Не удалось извлечь время из файла {fpath}")
            
            # Преобразуем строки в числа
            float_vals = [float(v) for v in values]
            
            # ВЫВОД ВСЕХ ЗНАЧЕНИЙ, по которым будет считаться среднее
            print(f"{basename}: значения = {float_vals}")
            
            # Вычисляем среднее арифметическое
            avg_val = np.mean(float_vals)
        
        sizes.append(n_val)
        avg_times.append(avg_val)
    
    order = np.argsort(sizes)
    return np.array(sizes)[order], np.array(avg_times)[order]

# Читаем данные
n_lapack, t_lapack = read_times(LAPACK_DIR, PREFIXES['LAPACK'])
n_mkl,    t_mkl    = read_times(MKL_DIR,    PREFIXES['MKL'])
n_numpy,  t_numpy  = read_times(NUMPY_DIR,  PREFIXES['NumPy'])

print("Загруженные данные (LAPACK):", list(zip(n_lapack, t_lapack)))
print("Загруженные данные (MKL):",    list(zip(n_mkl, t_mkl)))
print("Загруженные данные (NumPy):",  list(zip(n_numpy, t_numpy)))

# ------------------------------------------------------------
# АППРОКСИМАЦИЯ
# ------------------------------------------------------------
log_n_lapack = np.log(n_lapack)
log_t_lapack = np.log(t_lapack)
log_n_mkl = np.log(n_mkl)
log_t_mkl = np.log(t_mkl)
log_n_numpy = np.log(n_numpy)
log_t_numpy = np.log(t_numpy)

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

plt.scatter(n_lapack, t_lapack, color='green', label='LAPACK', zorder=5)
plt.scatter(n_mkl,    t_mkl,    color='purple', marker='s', label='MKL', zorder=5)
plt.scatter(n_numpy,  t_numpy,  color='blue', marker='^', label='NumPy, zorder=5)

n_fit = np.linspace(2000, 21000, 500)
plt.plot(n_fit, a_lapack * n_fit**c_lapack, '--', color='green')
plt.plot(n_fit, a_mkl    * n_fit**c_mkl,    '--', color='purple')
plt.plot(n_fit, a_numpy  * n_fit**c_numpy,  '--', color='blue')

for x, y1, y2, y3 in zip(n_lapack, t_lapack, t_mkl, t_numpy):
    plt.text(x, y1*1.4, f'{y1:.1f}s', ha='center', va='bottom',
             color='green', fontsize=8, fontweight='bold')
    plt.text(x, y2*0.6, f'{y2:.1f}s', ha='center', va='top',
             color='purple', fontsize=8, fontweight='bold')
    plt.text(x, y3*1.4, f'{y3:.1f}s', ha='center', va='bottom',
             color='blue', fontsize=8, fontweight='bold')

plt.xlabel('Размер матрицы ($n$)', fontsize=12, labelpad=10)
plt.ylabel('Время (с)', fontsize=12, labelpad=10)
plt.title('Аппроксимация времени LU-обращения (линейная регрессия в log-log),
          fontsize=14, pad=15)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5, which='both')

plt.tight_layout()
plt.show()
