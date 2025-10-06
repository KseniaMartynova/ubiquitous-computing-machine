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

def invert_matrix_with_svd(matrix):
    """
    Обращение матрицы через SVD-разложение
    """
    # Выполняем SVD: A = U * S * V^T
    U, S, Vt = np.linalg.svd(matrix, full_matrices=True)
    
    # Определяем порог для сингулярных значений
    eps = np.finfo(float).eps
    threshold = np.max(S) * max(matrix.shape) * eps
    
    # Инвертируем ненулевые сингулярные значения
    S_inv = np.zeros_like(S)
    S_inv[S > threshold] = 1.0 / S[S > threshold]
    
    # Создаем диагональную матрицу из инвертированных сингулярных значений
    S_inv_mat = np.zeros(matrix.shape)
    np.fill_diagonal(S_inv_mat, S_inv)
    
    # Вычисляем A^(-1) = V * S^(-1) * U^T
    A_inv = Vt.T @ S_inv_mat @ U.T
    
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
        print("Usage: python svd.py <matrix_size>")
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
    
    # Засекаем время SVD разложения и обращения
    start_time_svd = time.time()
    inverted_matrix = invert_matrix_with_svd(matrix)
    end_time_svd = time.time()
    svd_time = end_time_svd - start_time_svd
    
    # Проверяем корректность обращения
    is_correct = check_inversion_correctness(matrix, inverted_matrix)
    
    # Выводим результаты
    # print(f"Matrix size: {n}x{n}")
    # print(f"Time: {svd_time:.6f} seconds")
    # print(f"Verification: {'PASSED' if is_correct else 'FAILED'}")

    print(f"{n},N/A,N/A,{svd_time:.6f},{'PASSED' if is_correct else 'FAILED'},N/A")
    
if __name__ == "__main__":
    main()
