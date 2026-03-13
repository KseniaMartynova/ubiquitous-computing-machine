#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <cmath>
#include <omp.h>

// Генерация положительно определенной матрицы (без изменений)
void generate_positive_definite_matrix(double* A, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double avg = (A[i*n + j] + A[j*n + i]) / 2.0;
            A[i*n + j] = A[j*n + i] = avg;
        }
        A[i*n + i] += n;
    }
}

// Чистое обращение матрицы через SVD (используется LAPACKE_dgesdd)
void svd_invert(const double* A, int n, double* A_inv) {
    // Создаём копию входной матрицы (SVD модифицирует исходную)
    std::vector<double> A_copy(A, A + n * n);
    
    std::vector<double> S(n);
    std::vector<double> U(n * n);
    std::vector<double> VT(n * n);
    // Для dgesdd не требуется массив superb

    // Используем LAPACKE_dgesdd (разделяй и властвуй) вместо dgesvd
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', n, n,
                              A_copy.data(), n, S.data(), U.data(), n,
                              VT.data(), n);
    if (info != 0) {
        throw std::runtime_error("SVD decomposition (dgesdd) failed");
    }

    double max_sv = *std::max_element(S.begin(), S.end());
    double threshold = max_sv * n * std::numeric_limits<double>::epsilon();

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        S[i] = (S[i] > threshold) ? 1.0 / S[i] : 0.0;
    }

    // SinvUT = S^{-1} * U^T
    std::vector<double> SinvUT(n * n, 0.0);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            SinvUT[i * n + j] = S[i] * U[j * n + i]; // U^T[i,j] = U[j,i]
        }
    }

    // A_inv = V * SinvUT  (здесь VT уже хранит V^T, поэтому транспонируем при умножении)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, n, n, 1.0, VT.data(), n, SinvUT.data(), n,
                0.0, A_inv, n);
}

// Проверка корректности (умножение A * A_inv)
bool check_inversion(const double* A, const double* A_inv, int n, double tol = 1e-10) {
    std::vector<double> product(n * n, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, A, n, A_inv, n, 0.0, product.data(), n);
    double max_error = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            double error = std::abs(product[i * n + j] - expected);
            max_error = std::max(max_error, error);
        }
    }
    return max_error < tol;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Matrix size must be positive" << std::endl;
        return 1;
    }

    // Настройка MKL
    int num_threads = mkl_get_max_threads();
    mkl_set_num_threads(num_threads);

    // Генерация исходной матрицы
    std::vector<double> A(n * n);
    generate_positive_definite_matrix(A.data(), n);

    // Сохраняем копию для проверки
    std::vector<double> A_original = A;
    std::vector<double> A_inv(n * n); // массив для обратной матрицы

    // ЗАМЕР ВРЕМЕНИ (только обращение)
    auto start = std::chrono::high_resolution_clock::now();
    svd_invert(A_original.data(), n, A_inv.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Проверка корректности (после замера)
    bool success = check_inversion(A_original.data(), A_inv.data(), n);

    // Вывод в формате, совместимом с вашим Python-кодом
    std::cout << n << ",N/A,N/A," << elapsed.count() << ","
              << (success ? "PASSED" : "FAILED") << "," << num_threads << std::endl;

    return 0;
}
