#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <cmath>
#include <omp.h>

// Генерация положительно определенной матрицы
void generate_positive_definite_matrix(double* A, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double avg = (A[i*n + j] + A[j*n + i]) / 2.0;
            A[i*n + j] = A[j*n + i] = avg;
        }
        A[i*n + i] += n;  // Усиление диагонали
    }
}

// Обращение матрицы через SVD
bool invert_matrix_via_svd(std::vector<double>& A, int n) {
    // Копия для исходной матрицы
    std::vector<double> A_copy = A;
    
    // Выделяем память для SVD
    std::vector<double> S(n);          // Сингулярные значения
    std::vector<double> U(n * n);      // U матрица
    std::vector<double> VT(n * n);     // V^T матрица
    std::vector<double> superb(n-1);   // Вспомогательный массив

    // Выполняем SVD разложение
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n, n, 
                             A.data(), n, S.data(), U.data(), n, 
                             VT.data(), n, superb.data());
    
    if (info > 0) {
        std::cerr << "SVD decomposition failed to converge" << std::endl;
        return false;
    }

    // Инвертируем сингулярные значения с контролем малых значений
    double max_sv = *std::max_element(S.begin(), S.end());
    double threshold = max_sv * n * std::numeric_limits<double>::epsilon();
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        S[i] = (S[i] > threshold) ? 1.0 / S[i] : 0.0;
    }

    // Создаем Σ^(-1) * U^T
    std::vector<double> SinvUT(n * n, 0.0);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            SinvUT[i * n + j] = S[i] * U[j * n + i];  // U[j, i] из-за транспонирования
        }
    }

    // A^(-1) = V * Σ^(-1) * U^T
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                n, n, n, 1.0, VT.data(), n, SinvUT.data(), n, 
                0.0, A.data(), n);

    // Проверка корректности обращения
    std::vector<double> result(n * n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, 1.0, A_copy.data(), n, A.data(), n, 
                0.0, result.data(), n);

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

    int n = std::atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Matrix size must be positive" << std::endl;
        return 1;
    }

    // Настраиваем MKL
    int num_threads = mkl_get_max_threads();
    mkl_set_num_threads(num_threads);

    // Создаем и инициализируем матрицу
    std::vector<double> A(n * n);
    generate_positive_definite_matrix(A.data(), n);

    // Засекаем время
    auto start = std::chrono::high_resolution_clock::now();

    // Обращаем матрицу
    bool success = invert_matrix_via_svd(A, n);

    // Замеряем время
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Выводим результаты
    // std::cout << "Matrix size: " << n << "x" << n << std::endl;
    // std::cout << "Time: " << elapsed.count() << " seconds" << std::endl;
    // std::cout << "Verification: " << (success ? "PASSED" : "FAILED") << std::endl;
    // std::cout << "Threads used: " << num_threads << std::endl;

    std::cout << n << ",N/A,N/A," << elapsed.count() << "," 
    << (success ? "PASSED" : "FAILED") << "," << num_threads << std::endl;

    return 0;
}
