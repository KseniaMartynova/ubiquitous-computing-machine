#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <omp.h>

// Функция для создания положительно определенной матрицы
void generate_positive_definite_matrix(double* A, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Генерируем случайную матрицу
    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }

    // Делаем матрицу симметричной
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i * n + j] = A[j * n + i] = (A[i * n + j] + A[j * n + i]) / 2.0;
        }
        // Добавляем к диагональным элементам для гарантии положительной определенности
        A[i * n + i] += n;
    }
}

// Функция для обращения матрицы методом Гаусса-Жордана
void gauss_jordan_inversion(double* A, double* A_inv, int n) {
    // Создаем единичную матрицу для правой части
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_inv[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Прямой ход метода Гаусса-Жордана
    for (int i = 0; i < n; ++i) {
        // Проверка на ненулевой диагональный элемент
        if (std::abs(A[i * n + i]) < 1e-10) {
            std::cerr << "Matrix is singular or poorly conditioned" << std::endl;
            return;
        }

        // Нормализация текущей строки
        double pivot = A[i * n + i];
        #pragma omp parallel for
        for (int j = 0; j < n; ++j) {
            A[i * n + j] /= pivot;
            A_inv[i * n + j] /= pivot;
        }

        // Обнуление элементов столбца
        #pragma omp parallel for
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = A[k * n + i];
                for (int j = 0; j < n; ++j) {
                    A[k * n + j] -= factor * A[i * n + j];
                    A_inv[k * n + j] -= factor * A_inv[i * n + j];
                }
            }
        }
    }
}

// Функция для проверки корректности обращения матрицы
bool check_inversion_correctness(const double* A, const double* A_inv, int n) {
    std::vector<double> result(n * n, 0.0);
    
    // Умножаем исходную матрицу на обратную
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, 1.0, A, n, A_inv, n, 0.0, result.data(), n);
    
    // Проверяем, близок ли результат к единичной матрице
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

    int n = std::stoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Matrix size must be positive" << std::endl;
        return 1;
    }

    // Устанавливаем количество потоков для OpenBLAS и OpenMP
    int num_threads = omp_get_max_threads();
    openblas_set_num_threads(num_threads);
    omp_set_num_threads(num_threads);

    // Выделяем память для матриц
    std::vector<double> A(n * n);
    std::vector<double> A_copy(n * n);
    std::vector<double> A_inv(n * n);

    // Генерируем положительно определенную матрицу
    generate_positive_definite_matrix(A.data(), n);
    A_copy = A; // Сохраняем копию для проверки

    // Засекаем время начала обращения
    auto start = std::chrono::high_resolution_clock::now();

    // Обращаем матрицу методом Гаусса-Жордана
    gauss_jordan_inversion(A.data(), A_inv.data(), n);

    // Засекаем время окончания обращения
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Проверяем корректность обращения
    bool is_correct = check_inversion_correctness(A_copy.data(), A_inv.data(), n);

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
