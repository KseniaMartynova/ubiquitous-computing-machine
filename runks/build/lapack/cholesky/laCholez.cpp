#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <sys/resource.h>
#include <string>
#include <sstream>
 //Факторизация Холецкого
std::vector<std::string> called_routines;

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
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::vector<double> matrix = create_positive_definite_matrix(n, n);
    std::vector<double> inverse_matrix = matrix;

    // Получаем фактическое число потоков
    int num_threads = openblas_get_num_threads();

    auto start = std::chrono::steady_clock::now();

    called_routines.push_back("dpotrf");
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, inverse_matrix.data(), n);
    if (info != 0) {
        std::cerr << "Error in Cholesky decomposition" << std::endl;
        return 1;
    }

    called_routines.push_back("dpotri");
    info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, inverse_matrix.data(), n);
    if (info != 0) {
        std::cerr << "Error in matrix inversion" << std::endl;
        return 1;
    }

    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            inverse_matrix[i * n + j] = inverse_matrix[j * n + i];

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long rss_kb = usage.ru_maxrss;

    double checksum = 0.0;
    for (double v : matrix)
        checksum += v;

    std::ostringstream routines_oss;
    for (size_t i = 0; i < called_routines.size(); ++i) {
        if (i) routines_oss << ',';
        routines_oss << called_routines[i];
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "RESULT_SECONDS=" << diff.count() << std::endl;
    std::cout << "DIAG_THREADS=openblas/libopenblas:" << num_threads << std::endl;
    std::cout << "DIAG_PEAK_RSS_KB=" << rss_kb << std::endl;
    std::cout << "DIAG_ROUTINES=" << routines_oss.str() << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "DIAG_CHECKSUM=" << checksum << std::endl;

    return 0;
}
