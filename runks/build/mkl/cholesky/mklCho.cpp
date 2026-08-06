#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib>    // std::atoi
#include <cmath>      // std::abs
#include <sys/resource.h>  // getrusage

// список для хранения вызванных LAPACK/BLAS-функций
std::vector<std::string> called_routines;


void generate_positive_definite_matrix(double* A, int n) {
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
        std::cerr << "Использование: " << argv[0] << " <размер матрицы>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);

    std::vector<double> A(n * n);
    std::vector<double> A_inv(n * n);

    generate_positive_definite_matrix(A.data(), n);

    // Получаем текущее число потоков MKL 
    int num_threads = mkl_get_max_threads();

    // Копируем исходную матрицу для обращения
    std::copy(A.begin(), A.end(), A_inv.begin());


    auto start = std::chrono::high_resolution_clock::now();

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

    // Восстанавливаем симметрию: копируем нижний треугольник в верхний
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            A_inv[i * n + j] = A_inv[j * n + i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;   // на Linux – килобайты

    // Контрольная сумма обратной матрицы 
    double checksum = 0.0;
    for (double v : A_inv) {
        checksum += v;
    }

    // Формируем строку routines 
    std::ostringstream routines_oss;
    for (size_t i = 0; i < called_routines.size(); ++i) {
        if (i) routines_oss << ',';
        routines_oss << called_routines[i];
    }


    std::cout << std::fixed << std::setprecision(9);
    std::cout << "RESULT_SECONDS=" << elapsed.count() << std::endl;

    std::cout << "DIAG_THREADS=mkl:" << num_threads << std::endl;
    std::cout << "DIAG_PEAK_RSS_KB=" << rss_kb << std::endl;
    std::cout << "DIAG_ROUTINES=" << routines_oss.str() << std::endl;

    std::cout << std::setprecision(6);
    std::cout << "DIAG_CHECKSUM=" << checksum << std::endl;

    return 0;
}
