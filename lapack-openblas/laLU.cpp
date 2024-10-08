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

// Функция для обращения матрицы с использованием LU-разложения
bool lu_inverse(std::vector<double>& A, int n) {
    std::vector<int> ipiv(n);

    // Выполняем LU-разложение
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "LU-разложение не удалось." << std::endl;
        return false;
    }

    // Обращаем матрицу с использованием LU-разложения
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Обращение матрицы не удалось." << std::endl;
        return false;
    }

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

    // Обращаем матрицу с использованием LU-разложения
    bool success = lu_inverse(matrix, n);

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
