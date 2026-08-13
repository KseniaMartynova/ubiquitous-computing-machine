#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sys/resource.h>

// список для основных вызовов LAPACK/BLAS
std::vector<std::string> called_routines;

// Создание симметричной положительно определённой матрицы
std::vector<double> create_spd_matrix(int n, int seed) {
    std::vector<double> A(n * n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double avg = (A[i*n + j] + A[j*n + i]) / 2.0;
            A[i*n + j] = A[j*n + i] = avg;
        }
        A[i*n + i] += n;
    }
    return A;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    const int n = std::stoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Matrix size must be positive" << std::endl;
        return 1;
    }

    int num_threads_blas = openblas_get_num_threads();

    std::vector<double> A = create_spd_matrix(n, n);
    std::vector<double> A_orig = A;  // копия для контрольной суммы

    std::vector<double> S(n);
    std::vector<double> U(n * n);
    std::vector<double> VT(n * n);
    std::vector<double> A_inv(n * n);  // результат выделен до таймера

    auto start = std::chrono::steady_clock::now();

    // SVD
    called_routines.push_back("dgesdd");
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', n, n,
                              A.data(), n, S.data(), U.data(), n,
                              VT.data(), n);
    if (info != 0) {
        std::cerr << "SVD failed: " << info << std::endl;
        return 1;
    }

    // Инвертирование сингулярных чисел с отсечением
    double max_sv = *std::max_element(S.begin(), S.end());
    double threshold = max_sv * n * std::numeric_limits<double>::epsilon();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        S[i] = (S[i] > threshold) ? 1.0 / S[i] : 0.0;

    // Масштабирование строк VT
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            VT[i * n + j] *= S[i];
        }
    }

    // сборка обратной матрицы
    called_routines.push_back("dgemm");
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
                n, n, n,
                1.0, VT.data(), n,
                U.data(), n,
                0.0, A_inv.data(), n);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_duration = end - start;

    // Пиковое потребление памяти
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;

    // Контрольная сумма исходной матрицы
    double checksum = 0.0;
    for (double v : A_orig) checksum += v;

    // Формируем строку routines
    std::ostringstream routines_oss;
    for (size_t i = 0; i < called_routines.size(); ++i) {
        if (i) routines_oss << ',';
        routines_oss << called_routines[i];
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "RESULT_SECONDS=" << total_duration.count() << std::endl;
    std::cout << "DIAG_THREADS=openblas/libopenblas:" << num_threads_blas << std::endl;
    std::cout << "DIAG_PEAK_RSS_KB=" << rss_kb << std::endl;
    std::cout << "DIAG_ROUTINES=" << routines_oss.str() << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "DIAG_CHECKSUM=" << checksum << std::endl;

    return 0;
}
