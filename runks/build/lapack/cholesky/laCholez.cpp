#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <thread>
#include <sys/resource.h>   // для getrusage

// Создание положительно определённой матрицы
std::vector<double> create_positive_definite_matrix(int n) {
    std::vector<double> matrix(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i * n + j] = dis(gen);

    // Симметризация
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            matrix[j * n + i] = matrix[i * n + j];

    // Обеспечение положительной определённости
    for (int i = 0; i < n; ++i)
        matrix[i * n + i] += n;

    return matrix;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::vector<double> matrix = create_positive_definite_matrix(n);
    std::vector<double> inverse_matrix = matrix;

    // Определяем и задаём количество потоков для OpenBLAS
    int num_threads = std::thread::hardware_concurrency();
    openblas_set_num_threads(num_threads);

    // Замер времени
    auto start = std::chrono::high_resolution_clock::now();

    // Разложение Холецкого
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, inverse_matrix.data(), n);
    if (info != 0) {
        std::cerr << "Error in Cholesky decomposition" << std::endl;
        return 1;
    }

    // Обращение матрицы
    info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, inverse_matrix.data(), n);
    if (info != 0) {
        std::cerr << "Error in matrix inversion" << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Заполняем верхний треугольник обратной матрицы 
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            inverse_matrix[i * n + j] = inverse_matrix[j * n + i];

    // Пиковое использование памяти (RSS в килобайтах)
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;   

    // Контрольная сумма элементов обратной матрицы
    double checksum = 0.0;
    for (double v : inverse_matrix)
        checksum += v;

    // Вывод
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "RESULT_SECONDS=" << diff.count() << std::endl;

    std::cout << "DIAG_THREADS=openblas/libopenblas:" << num_threads << std::endl;
    std::cout << "DIAG_PEAK_RSS_KB=" << rss_kb << std::endl;
    std::cout << "DIAG_ROUTINES=dpotrf,dpotri" << std::endl;

    std::cout << std::setprecision(6);
    std::cout << "DIAG_CHECKSUM=" << checksum << std::endl;

    return 0;
}
