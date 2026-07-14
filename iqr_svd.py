import re
import os
import numpy as np
import matplotlib.pyplot as plt

def parse_times_robust(filepath, discard_first=True):
    times = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r':\s*([\d.]+)\s*s', line)
            if match:
                times.append(float(match.group(1)))
    if not times:
        raise ValueError(f"Нет данных в {filepath}")
    if discard_first and len(times) > 1:
        times = times[1:]
    median = np.median(times)
    q1 = np.percentile(times, 25)
    q3 = np.percentile(times, 75)
    return median, q1, q3, q3 - q1

# Размеры матриц
sizes = [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LAPACK_DIR = os.path.join(BASE_DIR, "runks", "build", "lapack", "svd", "results")
MKL_DIR    = os.path.join(BASE_DIR, "runks", "build", "mkl", "svd", "results")
NUMPY_DIR  = os.path.join(BASE_DIR, "runks", "build", "numpy", "svd", "results")


data = { 'LAPACK': {'sizes': [], 'medians': [], 'q1': [], 'q3': []},
         'MKL':    {'sizes': [], 'medians': [], 'q1': [], 'q3': []},
         'NumPy':  {'sizes': [], 'medians': [], 'q1': [], 'q3': []} }

# Сопоставление библиотек с их директориями и префиксами имён файлов
lib_config = {
    'LAPACK': (LAPACK_DIR, 'lapack'),
    'MKL':    (MKL_DIR, 'mkl'),
    'NumPy':  (NUMPY_DIR, 'num')   
}

for size in sizes:
    for lib, (directory, prefix) in lib_config.items():

        filepath = os.path.join(directory, f"{prefix}_svd_size_{size}.txt")
        try:
            median, q1, q3, _ = parse_times_robust(filepath)
            data[lib]['sizes'].append(size)
            data[lib]['medians'].append(median)
            data[lib]['q1'].append(q1)
            data[lib]['q3'].append(q3)
            print(f"{lib} {size}: med={median:.4f}, q1={q1:.4f}, q3={q3:.4f}")
        except Exception as e:
            print(f"Ошибка {lib} {size}: {e}")

# Построение графика
plt.figure(figsize=(12, 8))
plt.xscale('linear')
plt.yscale('log')

plt.yticks(
    ticks=[0.1, 0.15,0.3,0.47,0.9,1.5,2.0, 3.0,  4,  5,6.6,10, 25,35, 54, 80, 100, 150, 250, 500, 750, 1000,1500],
    labels=['0.1', '0.15','0.3','0.47','0.9','1.5','2.0', '3.0',  '4',  '5','6.6','10', '25','35', '54', '80', '100', '150', '250', '500', '750', '1000', '1500'],
    fontsize=10
)

plt.xlabel('Размер матрицы (n)', fontsize=12)
plt.ylabel('Время (с)', fontsize=12)
plt.title('Медиана и межквартильный размах (IQR) для Псевдообращения SVD', fontsize=14)

colors = {'LAPACK': 'green', 'MKL': 'purple', 'NumPy': 'blue'}
markers = {'LAPACK': 'o', 'MKL': 's', 'NumPy': '^'}

for lib in ['LAPACK', 'MKL', 'NumPy']:
    if data[lib]['sizes']:
        # Заливка IQR
        plt.fill_between(data[lib]['sizes'], data[lib]['q1'], data[lib]['q3'],
                         color=colors[lib], alpha=0.25)
        # Линия медиан
        plt.plot(data[lib]['sizes'], data[lib]['medians'],
                 color=colors[lib], marker=markers[lib], linestyle='-',
                 linewidth=2, markersize=8, label=lib)
        
        # Подписи значений медиан
        for x, y in zip(data[lib]['sizes'], data[lib]['medians']):
            y_offset = y * 1.1  # смещение для логарифмической шкалы
            plt.text(x, y_offset, f'{y:.2f}', ha='center', va='bottom',
                     fontsize=7, color=colors[lib], fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.5, which='both')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
