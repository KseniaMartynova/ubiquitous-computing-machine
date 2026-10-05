import numpy as np
import time
import sys
import resource
import os

# список вызванных BLAS-функций
called_routines = []

def get_blas_info():
    # Возвращает строку для DIAG_THREADS с перечислением всех обнаруженных бэкендов
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        entries = [] # Собираем все пулы, у которых есть информация о потоках
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
    # Генерация SPD-матрицы с фиксированным seed
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    A = 0.5 * (A + A.T)
    A += n * np.eye(n)
    return A


def compute_mult_residual(A, B, C, n):
    residual = 0.0
    for k in range(100):
        i = (k * 7) % n
        j = (k * 13) % n
        ref = float(np.dot(A[i, :], B[:, j]))
        diff = abs(C[i, j] - ref) / (abs(ref) + 1.0)
        if diff > residual:
            residual = diff
    return residual


def run_validate(n):
    A = generate_positive_definite_matrix(n, n)
    B = generate_positive_definite_matrix(n, n + 1)
    try:
        C = np.matmul(A, B)
    except Exception:
        print("VALIDATE_RESIDUAL=nan")
        print("VALIDATE_STATUS=fail")
        return 1
    residual = compute_mult_residual(A, B, C, n)
    ok = np.isfinite(residual) and residual <= 1e-10
    print(f"VALIDATE_RESIDUAL={residual:.6e}")
    print(f"VALIDATE_STATUS={'ok' if ok else 'fail'}")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]

    if len(args) == 2 and args[0] == "--validate":
        try:
            n = int(args[1])
            if n <= 0:
                raise ValueError
        except ValueError:
            print("Matrix size must be a positive integer")
            return 1
        return run_validate(n)

    if len(args) != 1:
        print("Usage: python multiply.py [--validate] <matrix_size>")
        sys.exit(1)

    try:
        n = int(args[0])
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Matrix size must be a positive integer")
        sys.exit(1)
# Генерируем две матрицы с разными seed
    matrix_a = generate_positive_definite_matrix(n, n)  # seed = n 
    matrix_b = generate_positive_definite_matrix(n, n + 1) # seed = n + 1
    called_routines.append('dgemm')

    start = time.perf_counter()
    result = np.matmul(matrix_a, matrix_b) # Умножение матриц
    elapsed = time.perf_counter() - start

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # Пиковая резидентная память (RSS) в КБ
    sum_a = float(np.sum(matrix_a)) # Контрольные суммы исходных матриц A и B
    sum_b = float(np.sum(matrix_b))
    diag_threads = get_blas_info() # Информация о BLAS/потоках
    routines_str = ','.join(called_routines) # Строка routines

    print(f"RESULT_SECONDS={elapsed:.9f}")
    print(f"DIAG_THREADS={diag_threads}")
    print(f"DIAG_PEAK_RSS_KB={rss_kb}")
    print(f"DIAG_ROUTINES={routines_str}")
    print(f"DIAG_CHECKSUM={sum_a:.6f},{sum_b:.6f}")


if __name__ == "__main__":
    sys.exit(main())
