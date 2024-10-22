#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <thread> // Для std::thread::hardware_concurrency

// Функция для генерации положительно определенной симметричной матрицы
void generate_positive_definite_matrix(double* A, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Генерация симметричной матрицы
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            A[i * n + j] = A[j * n + i] = dis(gen);
        }
    }

    // Добавляем n к диагональным элементам, чтобы гарантировать положительную определенность
    for (int i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Использование: " << argv[0] << " <размер матрицы>" << std::endl;
        return 1;
    }

    // Парсинг аргумента командной строки для получения размера матрицы
    int n;
    try {
        n = std::stoi(argv[1]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Неверный аргумент, ожидается целое число." << std::endl;
        return 1;
    }

    // Установка количества потоков для MKL
    mkl_set_num_threads(std::thread::hardware_concurrency());

    // Создаем матрицы A и A_inv
    std::vector<double> A(n * n);
    std::vector<double> A_inv(n * n);

    // Генерация положительно определенной матрицы
    generate_positive_definite_matrix(A.data(), n);

    // Копируем A в A_inv для обращения
    std::copy(A.begin(), A.end(), A_inv.begin());

    // Замер времени начала обращения
    auto start = std::chrono::high_resolution_clock::now();

    // Cholesky-разложение
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, A_inv.data(), n);
    if (info != 0) {
        std::cerr << "Ошибка в LAPACKE_dpotrf: " << info << std::endl;
        return 1;
    }

    // Вычисление обратной матрицы на основе разложения Холеcки
    info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, A_inv.data(), n);
    if (info != 0) {
        std::cerr << "Ошибка в LAPACKE_dpotri: " << info << std::endl;
        return 1;
    }

    // Замер времени окончания обращения
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Время, затраченное на обращение матрицы размерности "
              << n << "x" << n << ": "
              << diff.count() << " секунд" << std::endl;

    return 0;
}

