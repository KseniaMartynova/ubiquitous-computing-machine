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
// список для хранения вызванных LAPACK/BLAS-функций
std::vector<std::string> called_routines;


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
    std::vector<double> A_inv(A, A + n * n);

    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, A_inv.data(), n);
    if (info != 0) {
        std::cout << "VALIDATE_RESIDUAL=nan\n";
        std::cout << "VALIDATE_STATUS=fail\n";
        return 1;
    }
    info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, A_inv.data(), n);
    if (info != 0) {
        std::cout << "VALIDATE_RESIDUAL=nan\n";
        std::cout << "VALIDATE_STATUS=fail\n";
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            A_inv[i * n + j] = A_inv[j * n + i];
        }
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
        std::cerr << "Использование: " << argv[0] << " [--validate] <размер матрицы>" << std::endl;
        return 1;
    }

    std::vector<double> A(n * n);
    generate_positive_definite_matrix(A.data(), n, n);

    if (validate) {
        return run_validate(A.data(), n);
    }

    
    std::vector<double> A_inv(n * n);
    int num_threads = mkl_get_max_threads();
    std::copy(A.begin(), A.end(), A_inv.begin());// Копируем исходную матрицу для обращения

    auto start = std::chrono::steady_clock::now();
// Факторизация Холецкого (нижний треугольник)
    called_routines.push_back("dpotrf");
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, A_inv.data(), n);
    if (info != 0) {
        std::cerr << "Ошибка при выполнении dpotrf: " << info << std::endl;
        return 1;
    }
// Обращение матрицы на основе разложения Холецкого
    called_routines.push_back("dpotri");
    info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, A_inv.data(), n);
    if (info != 0) {
        std::cerr << "Ошибка при выполнении dpotri: " << info << std::endl;
        return 1;
    }

    for (int i = 0; i < n; ++i) {// Восстанавливаем симметрию: копируем нижний треугольник в верхний
        for (int j = i + 1; j < n; ++j) {
            A_inv[i * n + j] = A_inv[j * n + i];
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;

    double checksum = 0.0;// Контрольная сумма
    for (double v : A) checksum += v;

    std::ostringstream routines_oss;// Формируем строку routines 
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
