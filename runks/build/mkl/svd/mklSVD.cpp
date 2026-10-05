#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <sys/resource.h>   // для getrusage
#include <stdexcept>        // для std::runtime_error

// список вызванных  LAPACK/BLAS
std::vector<std::string> called_routines;

// Генерация симметричной положительно определённой матрицы
void generate_spd_matrix(double* A, int n, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    // Заполняем случайными числами
    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }
    // Симметризация усреднением и добавление n к диагонали
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double avg = (A[i*n + j] + A[j*n + i]) / 2.0;
            A[i*n + j] = A[j*n + i] = avg;
        }
        A[i*n + i] += n;
    }
}

// Обращение матрицы через SVD
void svd_invert(double* A, int n, double* A_inv,
                std::vector<double>& S,
                std::vector<double>& U,
                std::vector<double>& VT,
                std::vector<double>& SinvUT) {
     // SVD (dgesdd)
    called_routines.push_back("dgesdd");
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', n, n,
                              A, n, S.data(), U.data(), n, VT.data(), n);
    if (info != 0) {
        throw std::runtime_error("SVD decomposition failed");
    }
    // Инвертирование сингулярных чисел
    double max_sv = *std::max_element(S.begin(), S.end());
    double threshold = max_sv * n * std::numeric_limits<double>::epsilon();

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        S[i] = (S[i] > threshold) ? 1.0 / S[i] : 0.0;
    }
    // Формирование S^{-1} * U^T
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            SinvUT[i * n + j] = S[i] * U[j * n + i];
        }
    }
    // сборка A_inv = V * (S^{-1} U^T)
    called_routines.push_back("dgemm");
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, n, n,
                1.0, VT.data(), n,
                SinvUT.data(), n,
                0.0, A_inv, n);
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
int run_validate(double* A, int n) {
    std::vector<double> A_copy(A, A + n * n);
    std::vector<double> A_inv(n * n);
    std::vector<double> S(n);
    std::vector<double> U(n * n);
    std::vector<double> VT(n * n);
    std::vector<double> SinvUT(n * n, 0.0);

    try {
        svd_invert(A_copy.data(), n, A_inv.data(), S, U, VT, SinvUT);
    } catch (const std::exception&) {
        std::cout << "VALIDATE_RESIDUAL=nan\n";
        std::cout << "VALIDATE_STATUS=fail\n";
        return 1;
    }

    double residual = compute_residual(A, A_inv.data(), n);
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

    std::vector<double> A(n * n);
    generate_spd_matrix(A.data(), n, n);// A – SPD матрица

    if (validate) {
        return run_validate(A.data(), n);
    }
    int num_threads = mkl_get_max_threads();

    std::vector<double> A_original = A;
    std::vector<double> A_inv(n * n);
    // Выделяем рабочие векторы до таймера
    std::vector<double> S(n);
    std::vector<double> U(n * n);
    std::vector<double> VT(n * n);
    std::vector<double> SinvUT(n * n, 0.0);

    auto start = std::chrono::steady_clock::now();
    svd_invert(A_original.data(), n, A_inv.data(), S, U, VT, SinvUT);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // Пиковая память (RSS)
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;
     // Контрольная сумма исходной матрицы
    double checksum = 0.0;
    for (double v : A) checksum += v;
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
    std::cout << "DIAG_CHECKSUM=" << checksum << std::endl;

    return 0;
}
