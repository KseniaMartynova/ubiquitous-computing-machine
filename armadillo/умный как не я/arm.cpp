#include <iostream>
#include <armadillo>
#include <chrono>
#include <cstdlib>

bool check_inversion_correctness(const arma::mat& A, const arma::mat& A_inv) {
    arma::mat result = A * A_inv;
    double eps = 1e-6;
    for (size_t i = 0; i < A.n_rows; ++i) {
        for (size_t j = 0; j < A.n_cols; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(result(i, j) - expected) > eps) {
                std::cerr << "Ошибка в элементе (" << i << ", " << j << "): " << result(i, j) << " != " << expected << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);

    if (n <= 0) {
        std::cerr << "Matrix size must be a positive integer." << std::endl;
        return 1;
    }

    arma::mat A = arma::randu<arma::mat>(n, n);
    A = A * A.t(); // Сделать матрицу положительно определенной

    // Обращение матрицы и замер времени
    auto start = std::chrono::high_resolution_clock::now();
    arma::mat A_inv = arma::inv(A);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Время, затраченное на обращение матрицы размерности "
              << n << "x" << n << ": "
              << diff.count() << " секунд" << std::endl;

    // Проверка корректности обращения матрицы
    if (check_inversion_correctness(A, A_inv)) {
        std::cout << "Обращение матрицы выполнено корректно." << std::endl;
    } else {
        std::cerr << "Обращение матрицы выполнено некорректно." << std::endl;
    }

    return 0;
}
