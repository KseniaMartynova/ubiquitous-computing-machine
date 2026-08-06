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

//  симметризация + сдвиг диагонали на n
std::vector<double> create_spd_matrix(int n) {
    std::vector<double> A(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Заполняем случайными числами
    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }

    // Симметризация (A = (A + A^T) / 2) и добавление n к диагонали
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double avg = (A[i*n + j] + A[j*n + i]) / 2.0;
            A[i*n + j] = A[j*n + i] = avg;
        }
        A[i*n + i] += n;
    }

    return A;
}


bool verify_inversion(const std::vector<double>& A, const std::vector<double>& A_inv, int n) {
    std::vector<double> result(n * n, 0.0);
    
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

    // Генерация исходной матрицы 
    std::vector<double> A = create_spd_matrix(n);
    std::vector<double> A_orig = A;  // копия для последующей проверки

    // SVD параметры
    std::vector<double> S(n);
    std::vector<double> U(n * n);
    std::vector<double> VT(n * n);
    
    // Рабочие массивы для dgesdd 
    int lwork = 4 * n + n * n;
    std::vector<double> work(lwork);
    std::vector<int> iwork(8 * n);
    
    auto svd_start = std::chrono::high_resolution_clock::now();
    
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', n, n,
                              A.data(), n, S.data(), U.data(), n,
                              VT.data(), n);
    
    auto svd_end = std::chrono::high_resolution_clock::now();

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

    // Сборка обратной матрицы: A_inv = V * S^{-1} * U^T = VT^T * U^T
    auto inv_start = std::chrono::high_resolution_clock::now();
    std::vector<double> A_inv(n * n);
    cblas_dgemm(
        CblasRowMajor, 
        CblasTrans,     // VT^T = V
        CblasTrans,     // U^T
        n, n, n, 
        1.0, 
        VT.data(), n, 
        U.data(), n, 
        0.0, 
        A_inv.data(), n
    );
    auto inv_end = std::chrono::high_resolution_clock::now();

    // Общее время: SVD + сборка обратной матрицы
    auto svd_duration = std::chrono::duration<double>(svd_end - svd_start);
    auto inv_duration = std::chrono::duration<double>(inv_end - inv_start);
    auto total_duration = svd_duration + inv_duration;

    bool is_correct = verify_inversion(A_orig, A_inv, n);

    std::cout << "Time to svd " << n << "x" << n << " matrices: " 
              << total_duration.count() << " s" << std::endl;

    return 0;
}
