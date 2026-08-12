import numpy as np
import time
import sys
import resource
import os
from scipy.linalg.lapack import dgetrf, dgetri

#список вызванных LAPACK/BLAS-функций
called_routines = []

def get_blas_info():
    """Возвращает строку для DIAG_THREADS с перечислением всех обнаруженных бэкендов."""
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        # Собираем все пулы, у которых есть информация о потоках
        entries = []
        for pool in pools:
            if 'internal_api' in pool and 'num_threads' in pool:
                lib = pool['internal_api']       
                prefix = pool.get('prefix', lib)    # fallback на lib
                nthreads = pool['num_threads']
                entries.append(f"{lib}/{prefix}:{nthreads}")
        if entries:
            # Сортируем 
            entries.sort()
            return ';'.join(entries)
    except ImportError:
        pass

def generate_positive_definite_matrix(n):
    """Генерация симметричной положительно определённой матрицы."""
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T)
    A += n * np.eye(n)
    return A

def invert_matrix_with_lu(matrix):
    """
    Обращение через LU-разложение: dgetrf + dgetri.
    Регистрирует использованные подпрограммы LAPACK.
    """
    n = matrix.shape[0]

    # LU-факторизация
    called_routines.append('dgetrf')
    lu, piv, info = dgetrf(matrix, overwrite_a=1)
    if info != 0:
        raise np.linalg.LinAlgError("LU factorization failed")

    # Обращение матрицы на основе LU
    called_routines.append('dgetri')
    inv, info = dgetri(lu, piv, overwrite_lu=1)
    if info != 0:
        raise np.linalg.LinAlgError("Inverse computation failed")

    return inv

def main():
    if len(sys.argv) != 2:
        print("Usage: python lu.py <matrix_size>")
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
    inverted_matrix = invert_matrix_with_lu(matrix)
    elapsed = time.perf_counter() - start

    # Пиковое потребление памяти (RSS) в КБ
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Контрольная сумма
    checksum = float(np.sum(matrix))

    # Информация о потоках
    diag_threads = get_blas_info()

    # Строка с подпрограммами 
    routines_str = ','.join(called_routines)

    print(f"RESULT_SECONDS={elapsed:.9f}")
    print(f"DIAG_THREADS={diag_threads}")
    print(f"DIAG_PEAK_RSS_KB={rss_kb}")
    print(f"DIAG_ROUTINES={routines_str}")
    print(f"DIAG_CHECKSUM={checksum:.6f}")

if __name__ == "__main__":
    main()
