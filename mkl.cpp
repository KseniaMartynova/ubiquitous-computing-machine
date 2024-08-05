#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mkl.h>

void generate_positive_definite_matrix(double* A, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i*n + j] = A[j*n + i] = (A[i*n + j] + A[j*n + i]) / 2.0;
        }
    }

    for (int i = 0; i < n; ++i) {
        A[i*n + i] += n;
    }
}

bool check_inversion_correctness(const std::vector<double>& A, const std::vector<double>& A_inv, int n) {
    std::vector<double> result(n * n, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A.data(), n, A_inv.data(), n, 0.0, result.data(), n);

    double eps = 1e-6;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(result[i * n + j] - expected) > eps) {
                std::cerr << "Ошибка в элементе (" << i << ", " << j << "): " << result[i * n + j] << " != " << expected << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    int n;
    std::cout << "Введите размерность матрицы: ";
    std::cin >> n;

    std::vector<double> A(n * n);
    std::vector<double> A_inv(n * n);

    generate_positive_definite_matrix(A.data(), n);

    // Обращение матрицы и замер времени
    std::copy(A.begin(), A.end(), A_inv.begin());
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<lapack_int> ipiv(n);
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Ошибка в LAPACKE_dgetrf: " << info << std::endl;
        return 1;
    }

    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Ошибка в LAPACKE_dgetri: " << info << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Время, затраченное на обращение матрицы размерности "
              << n << "x" << n << ": "
              << diff.count() << " секунд" << std::endl;

    // Проверка корректности обращения матрицы
    if (check_inversion_correctness(A, A_inv, n)) {
        std::cout << "Обращение матрицы выполнено корректно." << std::endl;
    } else {
        std::cerr << "Обращение матрицы выполнено некорректно." << std::endl;
    }

    return 0;
}
