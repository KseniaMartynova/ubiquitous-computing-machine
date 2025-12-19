#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <cmath>

// Генерация положительно определенной матрицы
void generate_positive_definite_matrix(double* A, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Заполняем случайными числами
    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }

    // Делаем матрицу симметричной
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i*n + j] = A[j*n + i] = (A[i*n + j] + A[j*n + i]) / 2.0;
        }
    }

    // Усиливаем диагональ для положительной определенности
    for (int i = 0; i < n; ++i) {
        A[i*n + i] += n;
    }
}

// Проверка корректности обращения
bool verify_inversion(const std::vector<double>& A, const std::vector<double>& A_inv, int n) {
    std::vector<double> result(n * n, 0.0);

    // A * A_inv должно быть близко к единичной матрице
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 
                1.0, A.data(), n, A_inv.data(), n, 0.0, result.data(), n);

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

    // Устанавливаем число потоков для MKL
    int num_threads = mkl_get_max_threads();
    mkl_set_num_threads(num_threads);

    // Создаем матрицу
    std::vector<double> A(n * n);
    std::vector<double> A_inv(n * n);

    // Генерируем положительно определенную матрицу
    generate_positive_definite_matrix(A.data(), n);
    
    // Копируем исходную матрицу
    A_inv = A;

    // Засекаем время
    auto start = std::chrono::high_resolution_clock::now();

    // Выполняем LU-разложение
    std::vector<lapack_int> ipiv(n);
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "LU decomposition failed with code: " << info << std::endl;
        return 1;
    }

    // Вычисляем обратную матрицу на основе LU-разложения
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Matrix inversion failed with code: " << info << std::endl;
        return 1;
    }

    // Замеряем время
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Проверяем результат
    bool is_correct = verify_inversion(A, A_inv, n);

    // Выводим результаты
    // std::cout << "Matrix size: " << n << "x" << n << std::endl;
    // std::cout << "Time: " << elapsed.count() << " seconds" << std::endl;
    // std::cout << "Verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;
    // std::cout << "Threads used: " << num_threads << std::endl;

    // std::cout << n << "," << elapsed.count() << "," 
    // << (success ? "PASSED" : "FAILED") << "," 
    // << num_threads << std::endl;

    std::cout << n << "," << elapsed.count() << "," 
    << (is_correct ? "PASSED" : "FAILED") << "," 
    << num_threads << std::endl;

    return 0;
}
