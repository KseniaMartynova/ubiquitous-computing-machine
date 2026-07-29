#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <cmath>
#include <omp.h>

// ГЕНЕРАТОР A = B * B^T + n*I ===
void generate_spd_matrix(double* A, int n) {
    // Случайная матрица B (n x n)
    std::vector<double> B(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        B[i] = dis(gen);
    }

    // A = B * B^T (row-major)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, n, n, 1.0, B.data(), n, B.data(), n, 0.0, A, n);

    // A += n*I
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

// Обращение через SVD 
void svd_invert(double* A, int n, double* A_inv) {
    // A будет испорчена, но у нас уже есть копия снаружи
    std::vector<double> S(n);
    std::vector<double> U(n * n);
    std::vector<double> VT(n * n);

    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', n, n,
                              A, n, S.data(), U.data(), n, VT.data(), n);
    if (info != 0) {
        throw std::runtime_error("SVD decomposition failed");
    }

    double max_sv = *std::max_element(S.begin(), S.end());
    double threshold = max_sv * n * std::numeric_limits<double>::epsilon();

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        S[i] = (S[i] > threshold) ? 1.0 / S[i] : 0.0;
    }

    std::vector<double> SinvUT(n * n, 0.0);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            SinvUT[i * n + j] = S[i] * U[j * n + i];
        }
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, n, n, 1.0, VT.data(), n, SinvUT.data(), n,
                0.0, A_inv, n);
}

// Проверка корректности 
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

    int num_threads = mkl_get_max_threads();
    mkl_set_num_threads(num_threads);

    std::vector<double> A(n * n);
    // Вызываем новый генератор
    generate_spd_matrix(A.data(), n);

    std::vector<double> A_original = A;
    std::vector<double> A_inv(n * n);

    auto start = std::chrono::high_resolution_clock::now();
    svd_invert(A_original.data(), n, A_inv.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    bool success = check_inversion(A.data(), A_inv.data(), n);

    std::cout << "Time to svd " << n << "x" << n << " matrices: " 
              << elapsed.count() << " s" << std::endl;

    return 0;
}
