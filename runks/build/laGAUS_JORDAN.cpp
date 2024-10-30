#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>

// Функция для создания положительно определенной матрицы
std::vector<double> create_positive_definite_matrix(int n) {
    std::vector<double> matrix(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Заполняем матрицу случайными числами
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i * n + j] = dis(gen);
        }
    }

    // Делаем матрицу симметричной
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            matrix[j * n + i] = matrix[i * n + j];
        }
    }

    // Добавляем к диагональным элементам, чтобы сделать матрицу положительно определенной
    for (int i = 0; i < n; ++i) {
        matrix[i * n + i] += n;
    }

    return matrix;
}

// Функция для обращения матрицы методом Гаусса-Жордана
bool gauss_jordan_inverse(std::vector<double>& A, int n) {
    std::vector<double> A_inv(n * n);

    // Создаем единичную матрицу
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_inv[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Прямой ход Гаусса-Жордана
    for (int i = 0; i < n; ++i) {
        // Проверка на ненулевой элемент на диагонали
        if (A[i * n + i] == 0) {
            std::cerr << "Матрица вырождена, обращение невозможно." << std::endl;
            return false;
        }

        // Нормализация строки
        double divisor = A[i * n + i];
        for (int j = 0; j < n; ++j) {
            A[i * n + j] /= divisor;
            A_inv[i * n + j] /= divisor;
        }

        // Обнуление столбца
        #pragma omp parallel for
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = A[k * n + i];
                for (int j = 0; j < n; ++j) {
                    A[k * n + j] -= factor * A[i * n + j];
                    A_inv[k * n + j] -= factor * A_inv[i * n + j];
                }
            }
        }
    }

    // Копируем обратную матрицу в исходную матрицу
    std::copy(A_inv.begin(), A_inv.end(), A.begin());

    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::vector<double> matrix = create_positive_definite_matrix(n);

    // Замеряем время
    auto start = std::chrono::high_resolution_clock::now();

    // Обращаем матрицу методом Гаусса-Жордана
    bool success = gauss_jordan_inverse(matrix, n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (success) {
        std::cout << "Time to invert " << n << "x" << n << " matrix: " 
                  << diff.count() << " seconds" << std::endl;
    } else {
        std::cerr << "Matrix inversion failed." << std::endl;
        return 1;
    }

    return 0;
}
