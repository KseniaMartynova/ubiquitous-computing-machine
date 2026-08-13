import numpy as np
import time
import sys
import resource
import os

#  список вызванных BLAS-функций
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


def generate_positive_definite_matrix(n, seed):
    """Генерация SPD-матрицы с фиксированным seed."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
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


    # Генерируем две матрицы с разными seed
    matrix_a = generate_positive_definite_matrix(n, n)      # seed = n
    matrix_b = generate_positive_definite_matrix(n, n + 1)  # seed = n + 1
    called_routines.append('dgemm')
    # Замер времени с высоким разрешением
    start = time.perf_counter()

    # Умножение матриц
    
    result = np.matmul(matrix_a, matrix_b)

    elapsed = time.perf_counter() - start

    # Пиковая резидентная память (RSS) в КБ
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Контрольная сумма результирующей матрицы
    sum_a = float(np.sum(matrix_a))
    sum_b = float(np.sum(matrix_b))

    # Информация о BLAS/потоках
    diag_threads = get_blas_info()

    # Строка routines
    routines_str = ','.join(called_routines)

    print(f"RESULT_SECONDS={elapsed:.9f}")
    print(f"DIAG_THREADS={diag_threads}")
    print(f"DIAG_PEAK_RSS_KB={rss_kb}")
    print(f"DIAG_ROUTINES={routines_str}")
    print(f"DIAG_CHECKSUM={sum_a:.6f},{sum_b:.6f}")

if __name__ == "__main__":
    main()
