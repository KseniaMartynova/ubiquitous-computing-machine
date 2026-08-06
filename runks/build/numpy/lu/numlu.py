import numpy as np
import time
import sys
import resource
import os
from scipy.linalg.lapack import dgetrf, dgetri

#список вызванных LAPACK/BLAS-функций
called_routines = []

def get_blas_info():
    """Определяет библиотеку BLAS/LAPACK и число потоков."""
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

    # Запасной вариант на основе переменных окружения
    lib = 'openblas' if 'OPENBLAS_NUM_THREADS' in os.environ else \
          'mkl' if 'MKL_NUM_THREADS' in os.environ else 'unknown'
    nthreads = int(os.environ.get('OPENBLAS_NUM_THREADS',
                                 os.environ.get('MKL_NUM_THREADS', 1)))
    return lib, nthreads

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
    checksum = float(np.sum(inverted_matrix))

    # Информация о потоках
    lib_name, num_threads = get_blas_info()
    diag_threads = f"{lib_name}:{num_threads}"

    # Строка с подпрограммами 
    routines_str = ','.join(called_routines)

    print(f"RESULT_SECONDS={elapsed:.9f}")
    print(f"DIAG_THREADS={diag_threads}")
    print(f"DIAG_PEAK_RSS_KB={rss_kb}")
    print(f"DIAG_ROUTINES={routines_str}")
    print(f"DIAG_CHECKSUM={checksum:.6f}")

if __name__ == "__main__":
    main()
