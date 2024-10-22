#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <cmath>

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
        // Добавляем к диагональным элементам, чтобы сделать матрицу положительно определенной
        matrix[i * n + i] += n;
    }
    return matrix;
}

// Функция для проверки корректности обращения матрицы
double verify_inverse(const std::vector<double>& original, const std::vector<double>& inverse, int n) {
    std::vector<double> product(n * n, 0.0);
    
    // Умножаем исходную матрицу на обратную
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, original.data(), n, inverse.data(), n,
                0.0, product.data(), n);
    
    // Проверяем, насколько результат близок к единичной матрице
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            max_diff = std::max(max_diff, std::abs(product[i * n + j] - expected));
        }
    }
    return max_diff;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::vector<double> original_matrix = create_positive_definite_matrix(n);
    std::vector<double> working_matrix = original_matrix;  // Копия для работы

    auto start = std::chrono::high_resolution_clock::now();

    // Выполняем разложение Холецкого
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, working_matrix.data(), n);
    if (info != 0) {
        std::cerr << "Error in Cholesky decomposition" << std::endl;
        return 1;
    }

    // Обращаем матрицу
    info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, working_matrix.data(), n);
    if (info != 0) {
        std::cerr << "Error in matrix inversion" << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Заполняем верхнюю треугольную часть обратной матрицы
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            working_matrix[i * n + j] = working_matrix[j * n + i];
        }
    }

    // Проверяем корректность обращения
    double error = verify_inverse(original_matrix, working_matrix, n);

    std::cout << "Time to invert " << n << "x" << n << " matrix: " 
              << diff.count() << " seconds" << std::endl;
    std::cout << "Maximum error in inverse verification: " << error << std::endl;

    return 0;
}
