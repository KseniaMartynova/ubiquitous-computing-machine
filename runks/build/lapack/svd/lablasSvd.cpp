#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>

// Функция для генерации симметричной положительно определенной матрицы
std::vector<double> create_spd_matrix(int n) {
    std::vector<double> B(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Генерируем случайную матрицу B с параллелизацией
    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) 
        B[i] = dis(gen);

    // Вычисляем A = B * B^T + n*I (row-major)
    std::vector<double> A(n * n, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                n, n, n, 1.0, B.data(), n, B.data(), n, 0.0, A.data(), n);
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) 
        A[i * n + i] += n;

    return A;
}

// Функция для проверки корректности обращения
bool verify_inversion(const std::vector<double>& A, const std::vector<double>& A_inv, int n) {
    std::vector<double> result(n * n, 0.0);
    
    // A * A_inv должно быть близко к единичной матрице
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, 1.0, A.data(), n, A_inv.data(), n, 0.0, result.data(), n);
    
    double max_error = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            double error = std::abs(result[i * n + j] - expected);
            max_error = std::max(max_error, error);
        }
    }
    
    return max_error < 1e-10;
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

    // Установка количества потоков для OpenBLAS и OpenMP
    // int num_threads = omp_get_max_threads();
    // openblas_set_num_threads(num_threads);
    // std::cout << "Using " << num_threads << " threads" << std::endl;

    // Генерация исходной матрицы
    std::vector<double> A = create_spd_matrix(n);
    std::vector<double> A_orig = A;  // Сохраняем оригинальную матрицу

    // SVD параметры
    std::vector<double> S(n);        // Сингулярные значения
    std::vector<double> U(n * n);    // Левые сингулярные векторы
    std::vector<double> VT(n * n);   // Правые сингулярные векторы (транспонированные)
    
    // Определяем оптимальный размер рабочего массива
    int lwork = -1;
    double lwork_query;
    int info = LAPACKE_dgesvd_work(
        LAPACK_ROW_MAJOR, 
        'A',    // Все левые сингулярные векторы
        'A',    // Все правые сингулярные векторы
        n, n, 
        A.data(), n, 
        S.data(), 
        U.data(), n, 
        VT.data(), n, 
        &lwork_query, 
        lwork
    );

    if (info != 0) {
        std::cerr << "LWORK query failed: " << info << std::endl;
        return 1;
    }

    lwork = static_cast<int>(lwork_query);
    std::vector<double> work(lwork);

    // Засекаем время для SVD-разложения
    auto svd_start = std::chrono::high_resolution_clock::now();
    
    // Выполняем SVD
    info = LAPACKE_dgesvd_work(
        LAPACK_ROW_MAJOR, 
        'A', 'A', 
        n, n, 
        A.data(), n, 
        S.data(), 
        U.data(), n, 
        VT.data(), n, 
        work.data(), 
        lwork
    );
    
    auto svd_end = std::chrono::high_resolution_clock::now();

    if (info != 0) {
        std::cerr << "SVD failed: " << info << std::endl;
        return 1;
    }

    // Определяем порог для отсечения малых сингулярных значений
    double max_sv = *std::max_element(S.begin(), S.end());
    double threshold = max_sv * n * std::numeric_limits<double>::epsilon();
    
    // Инвертируем сингулярные значения с параллелизацией
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) 
        S[i] = (S[i] > threshold) ? 1.0 / S[i] : 0.0;

    // Масштабируем строки VT (столбцы V) на 1/S[i]
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            VT[i * n + j] *= S[i];
        }
    }

    // Вычисляем A_inv = V * S^{-1} * U^T = VT^T * U^T
    auto inv_start = std::chrono::high_resolution_clock::now();
    std::vector<double> A_inv(n * n);
    cblas_dgemm(
        CblasRowMajor, 
        CblasTrans,     // VT^T = V
        CblasTrans,     // U^T
        n, n, n, 
        1.0, 
        VT.data(), n,   // V (row-major)
        U.data(), n,    // U^T (row-major)
        0.0, 
        A_inv.data(), n
    );
    auto inv_end = std::chrono::high_resolution_clock::now();

    // Вычисляем время выполнения
    auto svd_duration = std::chrono::duration<double>(svd_end - svd_start);
    auto inv_duration = std::chrono::duration<double>(inv_end - inv_start);
    auto total_duration = svd_duration + inv_duration;

    // Проверяем корректность обращения
    bool is_correct = verify_inversion(A_orig, A_inv, n);

    // Выводим результаты
    // std::cout << "Matrix size: " << n << "x" << n << std::endl;
    // std::cout << "SVD time: " << svd_duration.count() << " seconds" << std::endl;
    // std::cout << "Inversion time: " << inv_duration.count() << " seconds" << std::endl;
    // std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;
    // std::cout << "Verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;

    std::cout << n << "," << svd_duration.count() << "," 
    << inv_duration.count() << "," << total_duration.count() << "," 
    // << (is_correct ? "PASSED" : "FAILED") << "," << num_threads << std::endl;
    << (is_correct ? "PASSED" : "FAILED") << "," << "N/A" << std::endl;

    return 0;
}
