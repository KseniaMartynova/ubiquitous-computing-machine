import numpy as np
import time
import sys

def generate_positive_definite_matrix(n):
    # Генерируем случайную матрицу
    A = np.random.rand(n, n)
    
    # Создаем симметричную матрицу
    A = 0.5 * (A + A.T)
    
    # Добавляем небольшое положительное число к диагонали для обеспечения положительной определенности
    A += n * np.eye(n)
    
    return A

def matrix_multiplication(matrix1, matrix2):
    n = matrix1.shape[0]
    result = np.zeros((n, n))  # Инициализируем результирующую матрицу нулями
    
    # Вручную умножаем матрицы
    for i in range(n):  # Проходим по строкам первой матрицы
        for j in range(n):  # Проходим по столбцам второй матрицы
            for k in range(n):  # Вычисляем скалярное произведение
                result[i, j] += matrix1[i, k] * matrix2[k, j]
    
    return result

def measure_time_for_matrix_multiplication(matrix1, matrix2):
    start_time = time.time()
    result_matrix = matrix_multiplication(matrix1, matrix2)
    end_time = time.time()
    return result_matrix, end_time - start_time

# Проверка наличия аргумента командной строки
if len(sys.argv) != 2:
    print("Использование: python script.py <размер матрицы>")
    sys.exit(1)

n = int(sys.argv[1])  # Размер матрицы

# Генерируем две положительно определенные матрицы
matrix1 = generate_positive_definite_matrix(n)
matrix2 = generate_positive_definite_matrix(n)

# Замер времени для умножения матриц
result_matrix, elapsed_time = measure_time_for_matrix_multiplication(matrix1, matrix2)
print(f"Время, затраченное на умножение матриц размерности {n}x{n}: {elapsed_time:.6f} секунд")
