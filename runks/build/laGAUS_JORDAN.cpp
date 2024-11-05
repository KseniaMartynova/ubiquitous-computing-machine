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

    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i * n + j] = A[j * n + i] = (A[i * n + j] + A[j * n + i]) / 2.0;
        }
        A[i * n + i] += n;
    }
}

// Функция для вывода матрицы
//void print_matrix(double* A, int n, const std::string& label) {
//    std::cout << label << ":\n";
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            std::cout << A[i * n + j] << " ";
//        }
//        std::cout << "\n";
//    }
//    std::cout << "\n";
//}

// Функция для обращения матрицы методом Гаусса-Жордана
void gauss_jordan_inversion(double* A, double* A_inv, int n) {
    // Создаем единичную матрицу
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_inv[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Выводим исходную матрицу
  //  print_matrix(A, n, "Исходная матрица");

    // Прямой ход Гаусса-Жордана
    for (int i = 0; i < n; ++i) {
        // Проверка на ненулевой элемент на диагонали
        if (A[i * n + i] == 0) {
            std::cerr << "Матрица вырождена, обращение невозможно." << std::endl;
            return;
        }

        // Нормализация строки
        double divisor = A[i * n + i];
        for (int j = 0; j < n; ++j) {
            A[i * n + j] /= divisor;
            A_inv[i * n + j] /= divisor;
        }

        // Вывод промежуточной матрицы после нормализации
    //    print_matrix(A, n, "Матрица после нормализации строки " + std::to_string(i));

        // Обнуление столбца
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

        // Вывод промежуточной матрицы после обнуления столбца
      //  print_matrix(A, n, "Матрица после обнуления столбца " + std::to_string(i));
    }
}

// Функция для проверки корректности обращения матрицы
bool check_inversion_correctness(double* A, double* A_inv, int n) {
    std::vector<double> result(n * n, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, A_inv, n, 0.0, result.data(), n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(result[i * n + j] - expected) > 1) {
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

    // Устанавливаем количество потоков для OpenBLAS
    int num_threads = omp_get_max_threads();
    openblas_set_num_threads(num_threads);

    std::vector<double> A(n * n);
    std::vector<double> A_inv(n * n);

    generate_positive_definite_matrix(A.data(), n);

    // Замеряем время
    auto start = std::chrono::high_resolution_clock::now();

    // Обращаем матрицу методом Гаусса-Жордана
    gauss_jordan_inversion(A.data(), A_inv.data(), n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Время обращения матрицы " << n << "x" << n 
              << " (метод Гаусса-Жордана): " << diff.count() << " секунд" << std::endl;
   // std::cout << "Использовано потоков: " << num_threads << std::endl;

    // Выводим обратную матрицу
   // print_matrix(A_inv.data(), n, "Обратная матрица");

    // Проверка корректности обращения матрицы
    if (check_inversion_correctness(A.data(), A_inv.data(), n)) {
        std::cout << "Проверка пройдена успешно" << std::endl;
    } else {
        std::cout << "Проверка не пройдена" << std::endl;
    }

    return 0;
}
