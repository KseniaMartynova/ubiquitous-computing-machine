import numpy as np
import time
import sys
import resource
import os
from scipy.linalg import inv

# список вызванных LAPACK/BLAS-функций
called_routines = []

def get_blas_info():
    # Возвращает строку для DIAG_THREADS с перечислением всех обнаруженных бэкендов
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
    # Генерация симметричной положительно определённой матрицы
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    A = 0.5 * (A + A.T)
    A += n * np.eye(n)
    return A

def invert_matrix_with_lu(matrix):
    called_routines.append('dgetrf')
    called_routines.append('dgetri')
    inverse = inv(matrix)
    return inverse

def compute_residual(A, A_inv):
    # Невязка ||A·A⁻¹ − I||_F / (||A||_F · ||A⁻¹||_F)
    n = A.shape[0]
    num = np.linalg.norm(A @ A_inv - np.eye(n), "fro")
    denom = np.linalg.norm(A, "fro") * np.linalg.norm(A_inv, "fro")
    if denom == 0:
        return float("nan")
    return num / denom

def run_validate(n):
    A = generate_positive_definite_matrix(n, n)
    try:
        A_inv = invert_matrix_with_lu(A.copy())
    except Exception:
        print("VALIDATE_RESIDUAL=nan")
        print("VALIDATE_STATUS=fail")
        return 1
    residual = compute_residual(A, A_inv)
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
        print("Usage: python lu.py [--validate] <matrix_size>")
        sys.exit(1)

    try:
        n = int(args[0])
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Matrix size must be a positive integer")
        sys.exit(1)

    matrix = generate_positive_definite_matrix(n, n)

    start = time.perf_counter()
    inverted_matrix = invert_matrix_with_lu(matrix)
    elapsed = time.perf_counter() - start

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # Пиковое потребление памяти (RSS) в КБ
    checksum = float(np.sum(matrix)) # Контрольная сумма исходной матрицы  
    diag_threads = get_blas_info() # Информация о потоках
    routines_str = ','.join(called_routines) # Строка с подпрограммами

    print(f"RESULT_SECONDS={elapsed:.9f}")
    print(f"DIAG_THREADS={diag_threads}")
    print(f"DIAG_PEAK_RSS_KB={rss_kb}")
    print(f"DIAG_ROUTINES={routines_str}")
    print(f"DIAG_CHECKSUM={checksum:.6f}")

if __name__ == "__main__":
    sys.exit(main())
