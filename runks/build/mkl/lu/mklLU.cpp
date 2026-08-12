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

// список вызванных подпрограмм LAPACK/BLAS
std::vector<std::string> called_routines;

// Генерация положительно определённой матрицы
void generate_positive_definite_matrix(double* A, int n, int seed) {
    std::random_device rd;
    std::mt19937 gen(rd());
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
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Matrix size must be positive" << std::endl;
        return 1;
    }

    // Получаем текущее число потоков MKL 
    int num_threads = mkl_get_max_threads();

    // Выделяем память и генерируем матрицу
    std::vector<double> A(n * n);
    std::vector<double> A_inv(n * n);
    generate_positive_definite_matrix(A.data(), n);
    A_inv = A;   // копия для обращения

    std::vector<lapack_int> ipiv(n);

    // Засекаем время
    auto start = std::chrono::steady_clock::now();

    // LU-разложение
    called_routines.push_back("dgetrf");
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "LU decomposition failed with code: " << info << std::endl;
        return 1;
    }

    // Обращение через LU
    called_routines.push_back("dgetri");
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Matrix inversion failed with code: " << info << std::endl;
        return 1;
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Пиковое потребление памяти
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;   

    // Контрольная сумма обратной матрицы
    double checksum = 0.0;
    for (double v : A) {
        checksum += v;
    }

    // Формируем строку DIAG_ROUTINES
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
