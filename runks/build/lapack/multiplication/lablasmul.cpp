#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <chrono>
#include <cblas.h>
#include <sys/resource.h>
#include <cmath>

// Список вызванных подпрограмм BLAS/LAPACK
std::vector<std::string> called_routines;

// Создание положительно определённой матрицы
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

// проверка умножения: 100 элементов считаются вручнуюи сравниваются с что вернул dgemm
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
    std::vector<double> A = create_positive_definite_matrix(n, n);
    std::vector<double> B = create_positive_definite_matrix(n, n + 1);
    std::vector<double> C(n * n, 0.0);

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
        n = std::stoi(argv[2]);
    } else if (argc == 2) {
        n = std::stoi(argv[1]);
    } else {
        std::cerr << "Usage: " << argv[0] << " [--validate] <matrix_size>" << std::endl;
        return 1;
    }

    if (validate) {
        return run_validate(n);
    }

    int num_threads = openblas_get_num_threads();

    std::vector<double> matrixA = create_positive_definite_matrix(n, n);
    std::vector<double> matrixB = create_positive_definite_matrix(n, n + 1);
    std::vector<double> result(n * n, 0.0);

    called_routines.push_back("dgemm");
    auto start = std::chrono::steady_clock::now();
// Регистрируем и выполняем умножение матриц
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0, matrixA.data(), n,
                matrixB.data(), n,
                0.0, result.data(), n);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
// Пиковое потребление памяти (RSS) в килобайтах
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;
// Контрольная сумма
    double sumA = 0.0, sumB = 0.0;
    for (double v : matrixA) sumA += v;
    for (double v : matrixB) sumB += v;
// Формируем строку routines
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
    std::cout << "DIAG_CHECKSUM=" << sumA << "," << sumB << std::endl;

    return 0;
}
