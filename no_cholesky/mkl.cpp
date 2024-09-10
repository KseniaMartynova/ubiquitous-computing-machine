#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mkl.h>

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

bool gauss_jordan_inverse(double* A, int n) {
    std::vector<double> A_inv(n * n);

    // Создаем единичную матрицу
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_inv[i*n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Прямой ход Гаусса-Жордана
    for (int i = 0; i < n; ++i) {
        // Проверка на ненулевой элемент на диагонали
        if (A[i*n + i] == 0) {
            std::cerr << "Матрица вырождена, обращение невозможно." << std::endl;
            return false;
        }

        // Нормализация строки
        double divisor = A[i*n + i];
        for (int j = 0; j < n; ++j) {
            A[i*n + j] /= divisor;
            A_inv[i*n + j] /= divisor;
        }

        // Обнуление столбца
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = A[k*n + i];
                for (int j = 0; j < n; ++j) {
                    A[k*n + j] -= factor * A[i*n + j];
                    A_inv[k*n + j] -= factor * A_inv[i*n + j];
                }
            }
        }
    }

    // Копируем обратную матрицу в исходную матрицу
    std::copy(A_inv.begin(), A_inv.end(), A);

    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Использование: " << argv[0] << " <размерность матрицы>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);

    if (n <= 0) {
        std::cerr << "Размерность матрицы должна быть положительным числом." << std::endl;
        return 1;
    }

    std::vector<double> A(n * n);

    generate_positive_definite_matrix(A.data(), n);

    // Обращение матрицы и замер времени
    auto start = std::chrono::high_resolution_clock::now();

    bool success = gauss_jordan_inverse(A.data(), n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (success) {
        std::cout << "Время, затраченное на обращение матрицы размерности "
                  << n << "x" << n << ": "
                  << diff.count() << " секунд" << std::endl;
    } else {
        std::cerr << "Обращение матрицы не удалось." << std::endl;
        return 1;
    }

    return 0;
}
