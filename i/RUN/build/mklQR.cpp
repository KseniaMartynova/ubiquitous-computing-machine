#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <thread> // Для std::thread::hardware_concurrency

// Функция для генерации случайной матрицы
void generate_random_matrix(double* A, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
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

    // Генерация случайной матрицы
    generate_random_matrix(A.data(), n);

    // Копируем A в A_inv для обращения
    std::copy(A.begin(), A.end(), A_inv.begin());

    // Переменные для хранения результатов разложения
    std::vector<double> tau(n);  // Вектор для хранения коэффициентов элементарных отражений

    // Замер времени начала обращения
    auto start = std::chrono::high_resolution_clock::now();

    // Выполнение QR-разложения
    int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n, tau.data());
    if (info != 0) {
        std::cerr << "Ошибка в LAPACKE_dgeqrf: " << info << std::endl;
        return 1;
    }

    // Восстановление матрицы Q из разложения QR
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, n, n, n, A_inv.data(), n, tau.data());
    if (info != 0) {
        std::cerr << "Ошибка в LAPACKE_dorgqr: " << info << std::endl;
        return 1;
    }

    // Вычисление обратной матрицы для R (R - верхнетреугольная матрица)
    info = LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', n, A_inv.data(), n);
    if (info != 0) {
        std::cerr << "Ошибка в LAPACKE_dtrtri: " << info << std::endl;
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

