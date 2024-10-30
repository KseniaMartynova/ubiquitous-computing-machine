#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <thread> // Для std::thread::hardware_concurrency

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

    // Добавляем к диагональным элементам, чтобы сделать матрицу положительно определенной
    for (int i = 0; i < n; ++i) {
        matrix[i * n + i] += n;
    }

    return matrix;
}

// Функция для обращения матрицы с использованием QR-разложения
bool qr_inverse(std::vector<double>& A, int n) {
    // Выполняем QR-разложение
    std::vector<double> tau(n);
    std::vector<int> jpvt(n, 0);
    int info = LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, n, n, A.data(), n, jpvt.data(), tau.data());
    if (info != 0) {
        std::cerr << "QR decomposition failed." << std::endl;
        return false;
    }

    // Обращаем верхнюю треугольную матрицу R
    std::vector<double> R_inv(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                R_inv[i * n + j] = 1.0 / A[i * n + j];
            } else if (i > j) {
                R_inv[i * n + j] = 0.0;
            } else {
                R_inv[i * n + j] = -A[i * n + j] / A[j * n + j];
                for (int k = i + 1; k < j; ++k) {
                    R_inv[i * n + j] -= A[i * n + k] * R_inv[k * n + j] / A[j * n + j];
                }
            }
        }
    }

    // Получаем обратную матрицу
    std::vector<double> A_inv(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_inv[i * n + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                A_inv[i * n + j] += R_inv[i * n + k] * A[k * n + j];
            }
        }
    }

    // Копируем обратную матрицу в исходную матрицу
    std::copy(A_inv.begin(), A_inv.end(), A.begin());

    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::vector<double> matrix = create_positive_definite_matrix(n);

    // Устанавливаем количество потоков для OpenBLAS
    int num_threads = std::thread::hardware_concurrency();
    std::cout << "Using " << num_threads << " threads for OpenBLAS." << std::endl;
    setenv("OPENBLAS_NUM_THREADS", std::to_string(num_threads).c_str(), 1);

    // Замеряем время
    auto start = std::chrono::high_resolution_clock::now();

    // Обращаем матрицу с использованием QR-разложения
    bool success = qr_inverse(matrix, n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (success) {
        std::cout << "Time to invert " << n << "x" << n << " matrix: " 
                  << diff.count() << " seconds" << std::endl;
    } else {
        std::cerr << "Matrix inversion failed." << std::endl;
        return 1;
    }

    return 0;
}
