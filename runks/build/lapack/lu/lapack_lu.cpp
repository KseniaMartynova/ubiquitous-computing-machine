#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <thread>
#include <cstdlib>
#include <cmath>

// Функция для создания положительно определенной матрицы
std::vector<double> create_positive_definite_matrix(int n) {
    std::vector<double> matrix(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Заполняем матрицу случайными числами
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i * n + j] = dis(gen);
        }
    }

    // Делаем матрицу симметричной
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            matrix[j * n + i] = matrix[i * n + j];
        }
    }

    // Добавляем к диагональным элементам для обеспечения положительной определенности
    for (int i = 0; i < n; ++i) {
        matrix[i * n + i] += n;
    }

    return matrix;
}

// Функция для проверки корректности обращения матрицы
bool check_inversion_result(const std::vector<double>& A, const std::vector<double>& A_inv, int n) {
    std::vector<double> result(n * n, 0.0);

    // Умножаем A на A_inv с использованием BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 
                1.0, A.data(), n, A_inv.data(), n, 0.0, result.data(), n);

    // Проверяем, близок ли результат к единичной матрице
    double tolerance = 1e-10;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(result[i * n + j] - expected) > tolerance) {
                return false;
            }
        }
    }

    return true;
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

    // Устанавливаем количество потоков для OpenBLAS
    int num_threads = std::thread::hardware_concurrency();
    openblas_set_num_threads(num_threads);

    // Создаем и инициализируем матрицу
    std::vector<double> A = create_positive_definite_matrix(n);
    std::vector<double> A_inv = A; // Копируем исходную матрицу для обращения

    // Начинаем отсчет времени
    auto start = std::chrono::high_resolution_clock::now();

    // Выполняем LU-разложение и обращение матрицы
    std::vector<lapack_int> ipiv(n);
    
    // LU факторизация
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "LU factorization failed with code: " << info << std::endl;
        return 1;
    }

    // Обращение матрицы с использованием LU факторизации
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Matrix inversion failed with code: " << info << std::endl;
        return 1;
    }

    // Останавливаем отсчет времени
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Проверяем результат
    bool is_correct = check_inversion_result(A, A_inv, n);

    // Выводим результаты
    // std::cout << "Matrix size: " << n << "x" << n << std::endl;
    // std::cout << "Time: " << elapsed.count() << " seconds" << std::endl;
    // std::cout << "Verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;
    // std::cout << "Threads used: " << num_threads << std::endl;

    std::cout << n << "," << elapsed.count() << "," 
    << (is_correct ? "PASSED" : "FAILED") << "," 
    << num_threads << std::endl;

    return 0;
}
