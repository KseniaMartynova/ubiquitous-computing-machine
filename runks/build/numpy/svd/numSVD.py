import numpy as np
import time
import sys
from scipy.linalg import svd

def generate_positive_definite_matrix(n):
    R = np.random.rand(n, n)
    A = 0.5 * (R + R.T)
    A += n * np.eye(n)
    return A

def measure_time_for_svd_inversion(matrix):
    start_time = time.time()
    U, s, Vt = svd(matrix, lapack_driver='gesdd')
    threshold = np.finfo(float).eps * max(matrix.shape) * np.max(s)
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    inverted = Vt.T @ np.diag(s_inv) @ U.T
    end_time = time.time()
    return inverted, end_time - start_time

def check_inversion_correctness(original_matrix, inverted_matrix):
    product = np.dot(original_matrix, inverted_matrix)
    identity_matrix = np.eye(original_matrix.shape[0])
    return np.allclose(product, identity_matrix)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Использование: python script <размер матрицы>")
        sys.exit(1)

    n = int(sys.argv[1])
    mat = generate_positive_definite_matrix(n)

    inverted, elapsed = measure_time_for_svd_inversion(mat)
    print(f"Time to svd {n}x{n} matrices: {elapsed:.6f} s")

    if not check_inversion_correctness(mat, inverted):
        print("Предупреждение: обратная матрица некорректна!")
