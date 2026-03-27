import numpy as np
import time
import sys

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

def invert_matrix_with_cholesky(matrix):
    """
    Обращение матрицы через разложение Холецкого
    """
    # Выполняем разложение Холецкого: A = L * L^T
    L = np.linalg.cholesky(matrix)
    
    # Обращаем нижнюю треугольную матрицу L
    L_inv = np.linalg.inv(L)
    
    # Вычисляем A^(-1) = (L * L^T)^(-1) = (L^T)^(-1) * L^(-1)
    A_inv = L_inv.T @ L_inv
    
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
        print("Usage: python cholesky.py <matrix_size>")
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
    
    # Обращаем матрицу с использованием разложения Холецкого
    inverted_matrix = invert_matrix_with_cholesky(matrix)
    
    # Засекаем время окончания
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Проверяем корректность обращения
    is_correct = check_inversion_correctness(matrix, inverted_matrix)
    
    # Выводим результаты
    # print(f"Matrix size: {n}x{n}")
    # print(f"Time: {elapsed_time:.6f} seconds")
    # print(f"Verification: {'PASSED' if is_correct else 'FAILED'}")

    print(f"{n},{elapsed_time:.6f},{'PASSED' if is_correct else 'FAILED'},N/A")
    
if __name__ == "__main__":
    main()
