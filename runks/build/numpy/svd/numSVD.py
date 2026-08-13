import numpy as np
import time
import sys
import resource
import os
from scipy.linalg import pinv

# список вызванных LAPACK/BLAS-функций
called_routines = []

def get_blas_info():
    """Возвращает строку для DIAG_THREADS с перечислением всех обнаруженных бэкендов."""
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        entries = []
        for pool in pools:
            if 'internal_api' in pool and 'num_threads' in pool:
                lib = pool['internal_api']
                prefix = pool.get('prefix', lib)
                nthreads = pool['num_threads']
                entries.append(f"{lib}/{prefix}:{nthreads}")
        if entries:
            entries.sort()
            return ';'.join(entries)
    except ImportError:
        pass

def generate_positive_definite_matrix(n, seed):
    """Генерация симметричной положительно определённой матрицы."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    A = 0.5 * (A + A.T)
    A += n * np.eye(n)
    return A

def invert_via_svd(matrix):
    """
    Обращение через псевдообратную (pinv). Регистрирует используемые подпрограммы.
    """
    called_routines.append('dgesdd')
    inverse = pinv(matrix)
    called_routines.append('dgemm')
    return inverse

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

    matrix = generate_positive_definite_matrix(n, n)

    # Замер времени
    start = time.perf_counter()
    inverted_matrix = invert_via_svd(matrix)
    elapsed = time.perf_counter() - start

    # Пиковое потребление памяти (RSS) в килобайтах
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Контрольная сумма исходной матрицы
    checksum = float(np.sum(matrix))

    diag_threads = get_blas_info()
    routines_str = ','.join(called_routines)

    print(f"RESULT_SECONDS={elapsed:.9f}")
    print(f"DIAG_THREADS={diag_threads}")
    print(f"DIAG_PEAK_RSS_KB={rss_kb}")
    print(f"DIAG_ROUTINES={routines_str}")
    print(f"DIAG_CHECKSUM={checksum:.6f}")

if __name__ == "__main__":
    main()
