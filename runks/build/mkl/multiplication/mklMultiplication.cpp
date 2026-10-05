#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <cmath>
#include <sys/resource.h>

// список вызванных подпрограмм BLAS/LAPACK
std::vector<std::string> called_routines;

// Генерация симметричной положительно определённой матрицы
void generate_positive_definite_matrix(double* A, int n, int seed) {
    std::mt19937 gen(seed);
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

double compute_mult_residual(const double* A, const double* B,
                             const double* C, int n) {
    double residual = 0.0;
    for (int k = 0; k < 100; ++k) {
        int i = (k * 7) % n;
        int j = (k * 13) % n;
        double ref = 0.0;
        for (int l = 0; l < n; ++l) {
            ref += A[i * n + l] * B[l * n + j];
        }
        double diff = std::abs(C[i * n + j] - ref) / (std::abs(ref) + 1.0);
        if (diff > residual) residual = diff;
    }
    return residual;
}
int run_validate(int n) {
    std::vector<double> A(n * n), B(n * n), C(n * n, 0.0);
    generate_positive_definite_matrix(A.data(), n, n);
    generate_positive_definite_matrix(B.data(), n, n + 1);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, A.data(), n, B.data(), n, 0.0, C.data(), n);

    double residual = compute_mult_residual(A.data(), B.data(), C.data(), n);
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
        std::cerr << "Использование: " << argv[0] << " [--validate] <размер матрицы>" << std::endl;
        return 1;
    }

    if (validate) {
        return run_validate(n);
    }

    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> C(n * n, 0.0);

    generate_positive_definite_matrix(A.data(), n, n);
    generate_positive_definite_matrix(B.data(), n, n + 1);

    int num_threads = mkl_get_max_threads();
    called_routines.push_back("dgemm");

    auto start = std::chrono::steady_clock::now();
// Регистрируем и выполняем умножение
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0, A.data(), n,
                B.data(), n,
                0.0, C.data(), n);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
// Пиковое потребление памяти
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;
// Контрольная сумма 
    double sumA = 0.0, sumB = 0.0;
    for (double v : A) sumA += v;
    for (double v : B) sumB += v;
// Формируем строку routines
    std::ostringstream routines_oss;
    for (size_t i = 0; i < called_routines.size(); ++i) {
        if (i) routines_oss << ',';
        routines_oss << called_routines[i];
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "RESULT_SECONDS=" << elapsed.count() << std::endl;
    std::cout << "DIAG_THREADS=mkl/libmkl_rt:" << num_threads << std::endl;
    std::cout << "DIAG_PEAK_RSS_KB=" << rss_kb << std::endl;
    std::cout << "DIAG_ROUTINES=" << routines_oss.str() << std::endl;

    std::cout << std::setprecision(6);
    std::cout << "DIAG_CHECKSUM=" << sumA << "," << sumB << std::endl;

    return 0;
}
