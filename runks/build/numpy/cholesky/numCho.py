import numpy as np
import sys
import time

def generate_spd_matrix(n):
    """Генерация положительно определенной симметричной матрицы"""
    A = np.random.rand(n, n)
    return A @ A.T + np.eye(n)*0.01  # Добавляем диагональное смещение для стабильности

def is_positive_definite(matrix):
    """Проверка положительной определенности через собственные значения"""
    return np.all(np.linalg.eigvalsh(matrix) > 0)

def main():
    try:
        n = int(sys.argv[1])
        if n <= 0:
            raise ValueError
    except (IndexError, ValueError):
        print("Usage: python cholesky.py <matrix_size>")
        sys.exit(1)

    # Генерация и проверка матрицы
    matrix = generate_spd_matrix(n)
    
    # Проверка симметричности
    symmetric = np.allclose(matrix, matrix.T)
    
    # Проверка положительной определенности
    pd_check = is_positive_definite(matrix)
    
    # Выполнение разложения
    start_time = time.time()
    try:
        L = np.linalg.cholesky(matrix)
        decomposition_success = True
    except np.linalg.LinAlgError:
        decomposition_success = False
    end_time = time.time()

    # Верификация результата
    verification = False
    if decomposition_success:
        reconstructed = L @ L.T
        verification = np.allclose(matrix, reconstructed, atol=1e-8)

    # Вывод результатов
    print(f"Time: {end_time - start_time:.6f}s")
    print(f"Symmetric: {'Yes' if symmetric else 'No'}")
    print(f"Positive definite: {'Yes' if pd_check else 'No'}")
    print(f"Decomposition: {'Success' if decomposition_success else 'Failed'}")
    print(f"Verification: {'Valid' if verification else 'Invalid'}")

if __name__ == "__main__":
    main()
