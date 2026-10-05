#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <cstdlib>
#include <cmath>
#include <sys/resource.h>

// LU-факторизация
std::vector<std::string> called_routines;

std::vector<double> create_positive_definite_matrix(int n, int seed) {
    std::vector<double> matrix(n * n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i * n + j] = dis(gen);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double avg = (matrix[i * n + j] + matrix[j * n + i]) / 2.0;
            matrix[i * n + j] = matrix[j * n + i] = avg;
        }
    }

    for (int i = 0; i < n; ++i)
        matrix[i * n + i] += n;

    return matrix;
}

// Возвращает невязку ||A·A⁻¹ − I||_F / (||A||_F · ||A⁻¹||_F)
double compute_residual(const double* A, const double* A_inv, int n) {
    std::vector<double> product(n * n, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, A, n, A_inv, n, 0.0, product.data(), n);

    double num_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            double d = product[i * n + j] - expected;
            num_sq += d * d;
        }
    }

    double norm_a_sq = 0.0, norm_inv_sq = 0.0;
    for (int i = 0; i < n * n; ++i) {
        norm_a_sq += A[i] * A[i];
        norm_inv_sq += A_inv[i] * A_inv[i];
    }
    double denom = std::sqrt(norm_a_sq) * std::sqrt(norm_inv_sq);
    if (denom == 0.0) return std::nan("");
    return std::sqrt(num_sq) / denom;
}

int run_validate(std::vector<double>& A, int n) {
    std::vector<double> A_inv = A;
    std::vector<lapack_int> ipiv(n);

    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cout << "VALIDATE_RESIDUAL=nan\n";
        std::cout << "VALIDATE_STATUS=fail\n";
        return 1;
    }
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cout << "VALIDATE_RESIDUAL=nan\n";
        std::cout << "VALIDATE_STATUS=fail\n";
        return 1;
    }

    double residual = compute_residual(A.data(), A_inv.data(), n);
    bool ok = std::isfinite(residual) && residual <= 1e-10;

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "VALIDATE_RESIDUAL=" << residual << "\n";
    std::cout << "VALIDATE_STATUS=" << (ok ? "ok" : "fail") << "\n";
    return ok ? 0 : 1;
}

int main(int argc, char* argv[]) {
    bool validate = false;
    int n = 0;

    if (argc == 3 && std::string(argv[1]) == "--validate") {
        validate = true;
        n = std::atoi(argv[2]);
    } else if (argc == 2) {
        n = std::atoi(argv[1]);
    } else {
        std::cerr << "Usage: " << argv[0] << " [--validate] <matrix_size>" << std::endl;
        return 1;
    }

    if (n <= 0) {
        std::cerr << "Matrix size must be positive" << std::endl;
        return 1;
    }

    std::vector<double> A = create_positive_definite_matrix(n, n);

    if (validate) {
        return run_validate(A, n);
    }

    int num_threads = openblas_get_num_threads();

    std::vector<double> A_inv = A;
    std::vector<lapack_int> ipiv(n);

    auto start = std::chrono::steady_clock::now();

    called_routines.push_back("dgetrf");
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "LU factorization failed with code: " << info << std::endl;
        return 1;
    }

    called_routines.push_back("dgetri");
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Matrix inversion failed with code: " << info << std::endl;
        return 1;
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;

    double checksum = 0.0;
    for (double v : A) checksum += v;

    std::ostringstream routines_oss;
    for (size_t i = 0; i < called_routines.size(); ++i) {
        if (i) routines_oss << ',';
        routines_oss << called_routines[i];
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "RESULT_SECONDS=" << elapsed.count() << std::endl;
    std::cout << "DIAG_THREADS=openblas/libopenblas:" << num_threads << std::endl;
    std::cout << "DIAG_PEAK_RSS_KB=" << rss_kb << std::endl;
    std::cout << "DIAG_ROUTINES=" << routines_oss.str() << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "DIAG_CHECKSUM=" << checksum << std::endl;

    return 0;
}
