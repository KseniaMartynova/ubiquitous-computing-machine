import numpy as np
import time
import sys
from scipy import linalg  # Adding scipy import for LU decomposition

def generate_positive_definite_matrix(n):
    """
    Генерация положительно определенной матрицы размера n x n
    """
    # Генерируем случайную матрицу
    A = np.random.rand(n, n)
    
    # Делаем матрицу симметричной
    A = 0.5 * (A + A.T)
    
    # Добавляем к диагонали для гарантии положительной определенности
    A += n * np.eye(n)
    
    return A

def invert_matrix_with_lu(matrix):
    """
    Обращение матрицы с использованием LU-разложения
    """
    # Используем scipy.linalg.lu вместо np.linalg.lu
    P, L, U = linalg.lu(matrix)
    
    n = matrix.shape[0]
    identity = np.eye(n)
    A_inv = np.zeros((n, n))
    
    # Решаем n систем линейных уравнений для каждого столбца единичной матрицы
    for i in range(n):
        # Решаем L * y = P.T * e_i для y методом прямой подстановки
        y = np.zeros(n)
        b = P.T @ identity[:, i]
        for j in range(n):
            y[j] = b[j] - L[j, :j] @ y[:j]
        
        # Решаем U * x = y для x методом обратной подстановки
        x = np.zeros(n)
        for j in range(n-1, -1, -1):
            x[j] = (y[j] - U[j, j+1:] @ x[j+1:]) / U[j, j]
        
        A_inv[:, i] = x
    
    return A_inv

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
    
    # Выводим результаты в том же формате, как в примерах
    print(f"{n},{elapsed_time:.6f},{'PASSED' if is_correct else 'FAILED'},N/A")
    
if __name__ == "__main__":
    main()
