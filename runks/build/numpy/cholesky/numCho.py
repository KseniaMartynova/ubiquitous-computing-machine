import numpy as np
import time
import sys
import resource
import os

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
    """Генерация случайной симметричной положительно определённой матрицы."""
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T)
    A += n * np.eye(n)
    return A

def invert_matrix_with_cholesky(matrix):
    """
    Обращение через разложение Холецкого (L L^T)^-1 = (L^-1)^T L^-1.
    Регистрирует используемые LAPACK/BLAS-функции.
    """
    # Разложение Холецкого -> dpotrf
    called_routines.append('dpotrf')
    L = np.linalg.cholesky(matrix)

    # Обращение нижней треугольной матрицы L -> dtrtri
    called_routines.append('dtrtri')
    L_inv = np.linalg.inv(L)

    # Сборка обратной матрицы -> dgemm
    called_routines.append('dgemm')
    A_inv = L_inv.T @ L_inv

    return A_inv

def main():
    if len(sys.argv) != 2:
        print("Usage: python cholesky.py <matrix_size>")
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
    inverted_matrix = invert_matrix_with_cholesky(matrix)
    elapsed = time.perf_counter() - start

    # Пиковое потребление памяти (RSS) в килобайтах
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Контрольная сумма обратной матрицы
    checksum = float(np.sum(inverted_matrix))

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
