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
    std::vector<double> inverse_matrix(n * n, 0.0);

    // Выделяем память для сингулярных значений и матриц U и V
    std::vector<double> singular_values(n);
    std::vector<double> U(n * n);
    std::vector<double> V(n * n);

    // Выделяем память для рабочего массива
    std::vector<double> work(3 * n);

    // Замеряем время
    auto start = std::chrono::high_resolution_clock::now();

    // Выполняем SVD разложение
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n, n, matrix.data(), n, singular_values.data(), U.data(), n, V.data(), n, work.data());
    if (info != 0) {
        std::cerr << "Error in SVD decomposition" << std::endl;
        return 1;
    }

    // Обращаем матрицу с использованием SVD
    for (int i = 0; i < n; ++i) {
        if (singular_values[i] > 1e-10) {
            singular_values[i] = 1.0 / singular_values[i];
        } else {
            singular_values[i] = 0.0;
        }
    }

    // Вычисляем обратную матрицу
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += U[i * n + k] * singular_values[k] * V[j * n + k];
            }
            inverse_matrix[i * n + j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Time to invert " << n << "x" << n << " matrix using SVD: " 
              << diff.count() << " seconds" << std::endl;

    return 0;
}
