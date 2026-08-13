#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib>      // std::atoi
#include <sys/resource.h> // getrusage

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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Использование: " << argv[0] << " <размер матрицы>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);

    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> C(n * n, 0.0);

    generate_positive_definite_matrix(A.data(), n, n);
    generate_positive_definite_matrix(B.data(), n, n + 1);

    // Получаем число потоков MKL 
    int num_threads = mkl_get_max_threads();
    called_routines.push_back("dgemm");
    // Засекаем время
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
    long rss_kb = usage.ru_maxrss;   // на Linux – килобайты

    // Контрольная сумма результирующей матрицы
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
