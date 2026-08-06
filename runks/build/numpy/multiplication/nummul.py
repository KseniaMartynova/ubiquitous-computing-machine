import numpy as np
import time
import sys
import resource
import os

#  список вызванных BLAS-функций
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

    # Запасной вариант по переменным окружения
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python multiply.py <matrix_size>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Matrix size must be a positive integer")
        sys.exit(1)

    # Генерируем две матрицы
    matrix_a = generate_positive_definite_matrix(n)
    matrix_b = generate_positive_definite_matrix(n)

    # Замер времени с высоким разрешением
    start = time.perf_counter()

    # Умножение матриц
    called_routines.append('dgemm')
    result = np.matmul(matrix_a, matrix_b)

    elapsed = time.perf_counter() - start

    # Пиковая резидентная память (RSS) в КБ
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Контрольная сумма результирующей матрицы
    checksum = float(np.sum(result))

    # Информация о BLAS/потоках
    lib_name, num_threads = get_blas_info()
    diag_threads = f"{lib_name}:{num_threads}"

    # Строка routines
    routines_str = ','.join(called_routines)

    print(f"RESULT_SECONDS={elapsed:.9f}")
    print(f"DIAG_THREADS={diag_threads}")
    print(f"DIAG_PEAK_RSS_KB={rss_kb}")
    print(f"DIAG_ROUTINES={routines_str}")
    print(f"DIAG_CHECKSUM={checksum:.6f}")

if __name__ == "__main__":
    main()
