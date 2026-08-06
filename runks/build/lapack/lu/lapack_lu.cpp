#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <sys/resource.h>   // для getrusage

// Функция для создания положительно определенной матрицы 
std::vector<double> create_positive_definite_matrix(int n) {
    std::vector<double> matrix(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i * n + j] = dis(gen);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            matrix[j * n + i] = matrix[i * n + j];

    for (int i = 0; i < n; ++i)
        matrix[i * n + i] += n;

    return matrix;
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

    int num_threads = std::thread::hardware_concurrency();
    openblas_set_num_threads(num_threads);

    std::vector<double> A = create_positive_definite_matrix(n);
    std::vector<double> A_inv = A; // копия для обращения

    std::vector<lapack_int> ipiv(n);

    auto start = std::chrono::high_resolution_clock::now();

    // LU-разложение
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "LU factorization failed with code: " << info << std::endl;
        return 1;
    }

    // Обращение через LU
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Matrix inversion failed with code: " << info << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Пиковое потребление памяти
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;   

    // Контрольная сумма обратной матрицы
    double checksum = 0.0;
    for (double v : A_inv) {
        checksum += v;
    }

    // 
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "RESULT_SECONDS=" << elapsed.count() << std::endl;

    std::cout << "DIAG_THREADS=openblas/libopenblas:" << num_threads << std::endl;
    std::cout << "DIAG_PEAK_RSS_KB=" << rss_kb << std::endl;
    std::cout << "DIAG_ROUTINES=dgetrf,dgetri" << std::endl;

    std::cout << std::setprecision(6);
    std::cout << "DIAG_CHECKSUM=" << checksum << std::endl;

    return 0;
}
