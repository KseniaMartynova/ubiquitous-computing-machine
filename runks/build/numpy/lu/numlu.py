import numpy as np
import time
import sys
from scipy.linalg.lapack import dgetrf, dgetri   # прямой доступ к LAPACK-функциям

def generate_positive_definite_matrix(n):
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T)
    A += n * np.eye(n)
    return A

def invert_matrix_with_lu(matrix):
    """
    Обращение матрицы через LU-разложение с использованием LAPACK (dgetrf + dgetri).
    Аналогично вызовам LAPACKE_dgetrf + LAPACKE_dgetri в C-тестах.
    """
    n = matrix.shape[0]
    # LU-факторизация, overwrite_a=1 разрешает перезапись входной матрицы для скорости
    lu, piv, info = dgetrf(matrix, overwrite_a=1)
    if info != 0:
        raise np.linalg.LinAlgError("LU factorization failed")
    # Вычисление обратной матрицы из LU-факторов (перезаписывает lu)
    inv, info = dgetri(lu, piv, overwrite_lu=1)
    if info != 0:
        raise np.linalg.LinAlgError("Inverse computation failed")
    return inv
def check_inversion_correctness(original_matrix, inverted_matrix):
    """
    Проверка корректности обращения матрицы
    """
    # Умножаем исходную матрицу на обратную
    product = original_matrix @ inverted_matrix
    
    # Вычисляем отклонение от единичной матрицы
    n = original_matrix.shape[0]
    identity = np.eye(n)
    error = np.max(np.abs(product - identity))
    
    return error < 1e-10

def main():
    """
    Основная функция для измерения времени обращения матрицы
    """
    if len(sys.argv) != 2:
        print("Usage: python lu.py <matrix_size>")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError("Matrix size must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Генерируем положительно определенную матрицу
    matrix = generate_positive_definite_matrix(n)
    
    # Засекаем время начала
    start_time = time.time()
    
    # Обращаем матрицу с использованием LU-разложения
    inverted_matrix = invert_matrix_with_lu(matrix)
    
    # Засекаем время окончания
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Проверяем корректность обращения
    is_correct = check_inversion_correctness(matrix, inverted_matrix)
    
    print(f"Time to lu {n}x{n} matrices: {elapsed_time:.6f} s")
    
if __name__ == "__main__":
    main()
