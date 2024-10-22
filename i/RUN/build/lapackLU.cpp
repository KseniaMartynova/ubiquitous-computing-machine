#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <cmath>

// Функция для создания случайной матрицы
std::vector<double> create_random_matrix(int n) {
    std::vector<double> matrix(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i * n + j] = dis(gen);
        }
        // Делаем диагональные элементы доминирующими для лучшей обусловленности
        matrix[i * n + i] = dis(gen) + n;
    }
    return matrix;
}

// Функция для проверки корректности обращения матрицы
double verify_inverse(const std::vector<double>& original, const std::vector<double>& inverse, int n) {
    std::vector<double> product(n * n, 0.0);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, original.data(), n, inverse.data(), n,
                0.0, product.data(), n);
    
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
    std::vector<double> original_matrix = create_random_matrix(n);
    std::vector<double> working_matrix = original_matrix;

    // Вектор для хранения перестановок
    std::vector<lapack_int> ipiv(n);

    auto start = std::chrono::high_resolution_clock::now();

    // LU-разложение
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, working_matrix.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Error in LU decomposition. Info: " << info << std::endl;
        return 1;
    }

    // Обращение матрицы на основе LU-разложения
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, working_matrix.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Error in matrix inversion. Info: " << info << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Проверяем корректность обращения
    double error = verify_inverse(original_matrix, working_matrix, n);

    // Вывод результатов
    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Time for LU decomposition and inversion: " 
              << diff.count() << " seconds" << std::endl;
    std::cout << "Maximum error in inverse verification: " << error << std::endl;

    return 0;
}
