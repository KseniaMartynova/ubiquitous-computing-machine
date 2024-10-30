import numpy as np
import time
import sys
from scipy.linalg import pinv

def generate_positive_definite_matrix(n):
    # Генерируем случайную матрицу
    A = np.random.rand(n, n)
    
    # Создаем симметричную матрицу
    A = 0.5 * (A + A.T)
    
    # Добавляем небольшое положительное число к диагонали для обеспечения положительной определенности
    A += n * np.eye(n)
    
    return A

def measure_time_for_svd_inversion(matrix):
    start_time = time.time()
    
    # Вычисляем псевдообратную матрицу с использованием scipy.linalg.pinv
    inverted_matrix = pinv(matrix)
    
    end_time = time.time()
    return inverted_matrix, end_time - start_time

def check_inversion_correctness(original_matrix, inverted_matrix):
    # Умножаем исходную матрицу на обратную
    product = np.dot(original_matrix, inverted_matrix)
    
    # Сравниваем с единичной матрицей
    identity_matrix = np.eye(original_matrix.shape[0])
    return np.allclose(product, identity_matrix)

# Проверка наличия аргумента командной строки
if len(sys.argv) != 2:
    print("Использование: python script.py <размер матрицы>")
    sys.exit(1)

n = int(sys.argv[1])  # Размер матрицы
positive_definite_matrix = generate_positive_definite_matrix(n)

# Замер времени для обращения матрицы с использованием SVD
inverted_matrix, elapsed_time = measure_time_for_svd_inversion(positive_definite_matrix)
print(f"Время, затраченное на обращение матрицы с использованием SVD: {elapsed_time:.6f} секунд")

# Проверка корректности обращения матрицы
is_correct = check_inversion_correctness(positive_definite_matrix, inverted_matrix)
print("Корректность обращения матрицы с использованием SVD:", is_correct)
