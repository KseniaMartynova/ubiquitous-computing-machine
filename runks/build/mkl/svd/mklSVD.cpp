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

// список вызванных ключевых подпрограмм LAPACK/BLAS
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

    std::vector<double> A(n * n);
    generate_spd_matrix(A.data(), n, n);   // A – SPD матрица
    std::vector<double> A_original = A;    // копия (можно не использовать)
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
    long rss_kb = usage.ru_maxrss;   // в килобайтах

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
