#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>

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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::vector<double> matrix = create_positive_definite_matrix(n);

    // Выделяем память для обратной матрицы
    std::vector<double> inverse_matrix = matrix;

    // Замеряем время
    auto start = std::chrono::high_resolution_clock::now();

    // Выполняем разложение Холецкого
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, inverse_matrix.data(), n);
    if (info != 0) {
        std::cerr << "Error in Cholesky decomposition" << std::endl;
        return 1;
    }

    // Обращаем матрицу
    info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, inverse_matrix.data(), n);
    if (info != 0) {
        std::cerr << "Error in matrix inversion" << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Заполняем верхнюю треугольную часть обратной матрицы
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            inverse_matrix[i * n + j] = inverse_matrix[j * n + i];
        }
    }

    std::cout << "Time to invert " << n << "x" << n << " matrix: " 
              << diff.count() << " seconds" << std::endl;

    return 0;
}
