import numpy as np
import time
import sys
import resource
import os
from scipy.linalg import svd

#писок вызванных LAPACK/BLAS-функций
called_routines = []

def get_blas_info():
    """Определяет библиотеку BLAS/LAPACK и число используемых потоков."""
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        for pool in pools:
            if pool['user_api'] == 'blas':
                lib = pool['internal_api']
                nthreads = pool['num_threads']
                return lib, nthreads
    except ImportError:
        pass

    # Запасной вариант – переменные окружения
    lib = 'openblas' if 'OPENBLAS_NUM_THREADS' in os.environ else \
          'mkl' if 'MKL_NUM_THREADS' in os.environ else 'unknown'
    nthreads = int(os.environ.get('OPENBLAS_NUM_THREADS',
                                 os.environ.get('MKL_NUM_THREADS', 1)))
    return lib, nthreads

def generate_positive_definite_matrix(n):
    """Симметричная положительно определённая матрица (SPD)."""
    R = np.random.rand(n, n)
    A = 0.5 * (R + R.T)
    A += n * np.eye(n)
    return A

def invert_via_svd(matrix):
    """
    Обращение матрицы через SVD (dgesdd) + матричные умножения (dgemm).
    Регистрирует использованные подпрограммы.
    """
    called_routines.append('dgesdd')
    U, s, Vt = svd(matrix, lapack_driver='gesdd')

    # Отсечение малых сингулярных чисел
    threshold = np.finfo(float).eps * max(matrix.shape) * np.max(s)
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)

    # A⁻¹ = V * S⁻¹ * Uᵀ
    # Реализовано через два матричных умножения с BLAS (dgemm)
    called_routines.append('dgemm')
    # Первое умножение: (V * diag(s_inv)) – эквивалентно масштабированию столбцов
    # Второе: результат @ U.T – фактически dgemm
    inverted = (Vt.T * s_inv) @ U.T   # broadcasting выполнит масштабирование, затем @ вызовет dgemm

    return inverted

def main():
    if len(sys.argv) != 2:
        print("Usage: python svd.py <matrix_size>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Matrix size must be a positive integer")
        sys.exit(1)

    matrix = generate_positive_definite_matrix(n)

    # Замер времени
    start = time.perf_counter()
    inverted_matrix = invert_via_svd(matrix)
    elapsed = time.perf_counter() - start

    # Пиковое потребление памяти (RSS) в килобайтах
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Контрольная сумма
    checksum = float(np.sum(inverted_matrix))

    # Информация о потоках и библиотеке BLAS
    lib_name, num_threads = get_blas_info()
    diag_threads = f"{lib_name}:{num_threads}"

    # Строка с вызванными подпрограммами
    routines_str = ','.join(called_routines)

    print(f"RESULT_SECONDS={elapsed:.9f}")
    print(f"DIAG_THREADS={diag_threads}")
    print(f"DIAG_PEAK_RSS_KB={rss_kb}")
    print(f"DIAG_ROUTINES={routines_str}")
    print(f"DIAG_CHECKSUM={checksum:.6f}")

if __name__ == "__main__":
    main()
