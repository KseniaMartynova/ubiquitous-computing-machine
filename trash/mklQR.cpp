#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib> // Для std::atoi
#include <cmath>   // Для std::abs

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

bool check_inversion_result(const std::vector<double>& A, const std::vector<double>& A_inv, int n) {
    std::vector<double> result(n * n, 0.0);

    // Умножение A на A_inv
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i * n + j] += A[i * n + k] * A_inv[k * n + j];
            }
        }
    }

    // Проверка на близость к единичной матрице
    double tolerance = 1e-6;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(result[i * n + j] - expected) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Использование: " << argv[0] << " <размер матрицы>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);

    std::vector<double> A(n * n);
    std::vector<double> A_inv(n * n);

    generate_positive_definite_matrix(A.data(), n);

    // Обращение матрицы и замер времени
    std::copy(A.begin(), A.end(), A_inv.begin());
    auto start = std::chrono::high_resolution_clock::now();

    // Выполнение QR-разложения
    std::vector<double> tau(n);
    int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n, tau.data());
    if (info != 0) {
        std::cerr << "Ошибка при выполнении dgeqrf: " << info << std::endl;
        return 1;
    }

    // Формирование ортогональной матрицы Q
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, n, n, n, A_inv.data(), n, tau.data());
    if (info != 0) {
        std::cerr << "Ошибка при выполнении dorgqr: " << info << std::endl;
        return 1;
    }

    // Обращение верхней треугольной матрицы R
    std::vector<double> R(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j >= i) {
                R[i * n + j] = A_inv[i * n + j];
            } else {
                R[i * n + j] = 0.0;
            }
        }
    }

    std::vector<double> R_inv(n * n);
    info = LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', n, R.data(), n);
    if (info != 0) {
        std::cerr << "Ошибка при выполнении dtrtri: " << info << std::endl;
        return 1;
    }

    // Умножение R_inv на Q^T
    std::vector<double> Q_T(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Q_T[i * n + j] = A_inv[j * n + i];
        }
    }

    std::vector<double> temp(n * n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, R_inv.data(), n, Q_T.data(), n, 0.0, temp.data(), n);

    // Копируем результат в A_inv
    std::copy(temp.begin(), temp.end(), A_inv.begin());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Время, затраченное на обращение матрицы размерности "
              << n << "x" << n << ": "
              << diff.count() << " секунд" << std::endl;


    return 0;
}
