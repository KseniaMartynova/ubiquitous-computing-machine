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

    // Выделяем память для tau (используется в QR разложении)
    std::vector<double> tau(n);

    auto start = std::chrono::high_resolution_clock::now();

    // QR-разложение
    int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, n, n, working_matrix.data(), n, tau.data());
    if (info != 0) {
        std::cerr << "Error in QR decomposition. Info: " << info << std::endl;
        return 1;
    }

    // Обращение матрицы на основе QR-разложения
    // Сначала нужно вычислить обратную матрицу R
    std::vector<double> inverse_matrix(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        inverse_matrix[i * n + i] = 1.0;  // Создаем единичную матрицу
    }

    // Решаем систему R * X = I
    info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', n, n, 
                          working_matrix.data(), n, inverse_matrix.data(), n);
    if (info != 0) {
        std::cerr << "Error in triangular solve. Info: " << info << std::endl;
        return 1;
    }

    // Применяем Q^T к результату
    info = LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'T', n, n, n,
                          working_matrix.data(), n, tau.data(), 
                          inverse_matrix.data(), n);
    if (info != 0) {
        std::cerr << "Error in applying Q^T. Info: " << info << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Проверяем корректность обращения
    double error = verify_inverse(original_matrix, inverse_matrix, n);

    // Вывод результатов
    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Time for QR decomposition and inversion: " 
              << diff.count() << " seconds" << std::endl;
    std::cout << "Maximum error in inverse verification: " << error << std::endl;

    return 0;
}
