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

def main():
    """
    Основная функция для измерения времени умножения матриц
    """
    if len(sys.argv) != 2:
        print("Usage: python multiply.py <matrix_size>")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError("Matrix size must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Генерируем две положительно определенные матрицы
    matrix_a = generate_positive_definite_matrix(n)
    matrix_b = generate_positive_definite_matrix(n)
    
    # Засекаем время начала
    start_time = time.time()
    
    # Умножаем матрицы с использованием NumPy
    result = np.matmul(matrix_a, matrix_b)
    
    # Засекаем время окончания
    end_time = time.time()
    elapsed_time = end_time - start_time
    

    print(f"{n},{elapsed_time:.6f},N/A,N/A")
    
if __name__ == "__main__":
    main()
