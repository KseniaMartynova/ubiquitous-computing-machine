#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib> // Для std::atoi
#include <cmath>   // Для std::abs

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

bool check_inversion_result(const std::vector<double>& A, const std::vector<double>& A_inv, int n) {
    std::vector<double> result(n * n, 0.0);

    // Умножение A на A_inv
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i * n + j] += A[i * n + k] * A_inv[k * n + j];
            }
        }
    }

    // Проверка на близость к единичной матрице
    double tolerance = 1e-6;
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
        std::cerr << "Использование: " << argv[0] << " <размер матрицы>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);

    std::vector<double> A(n * n);
    std::vector<double> A_inv(n * n);

    generate_positive_definite_matrix(A.data(), n);

    // Выделяем память для сингулярных значений и матриц U и V
    std::vector<double> singular_values(n);
    std::vector<double> U(n * n);
    std::vector<double> V(n * n);

    // Выделяем память для рабочего массива
    std::vector<double> work(3 * n);

    // Обращение матрицы и замер времени
    auto start = std::chrono::high_resolution_clock::now();

    // Выполняем SVD разложение
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n, n, A.data(), n, singular_values.data(), U.data(), n, V.data(), n, work.data());
    if (info != 0) {
        std::cerr << "Ошибка в SVD разложении" << std::endl;
        return 1;
    }

    // Обращаем матрицу с использованием SVD
    for (int i = 0; i < n; ++i) {
        if (singular_values[i] > 1e-10) {
            singular_values[i] = 1.0 / singular_values[i];
        } else {
            singular_values[i] = 0.0;
        }
    }

    // Вычисляем обратную матрицу
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += U[i * n + k] * singular_values[k] * V[j * n + k];
            }
            A_inv[i * n + j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Время, затраченное на обращение матрицы размерности "
              << n << "x" << n << " с использованием SVD: "
              << diff.count() << " секунд" << std::endl;

    // Проверка результата
    if (check_inversion_result(A, A_inv, n)) {
        std::cout << "Обращение матрицы прошло правильно." << std::endl;
    } else {
        std::cout << "Обращение матрицы прошло неправильно." << std::endl;
    }

    return 0;
}
