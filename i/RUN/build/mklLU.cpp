#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <thread> // Для std::thread::hardware_concurrenc

void generate_positive_definite_matrix(double* A, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            A[i*n + j] = A[j*n + i] = dis(gen);
        }
    }

    for (int i = 0; i < n; ++i) {
        A[i*n + i] += n;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Использование: " << argv[0] << " <размер матрицы>" << std::endl;
        return 1;
    }

    int n;
    try {
        n = std::stoi(argv[1]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Неверный аргумент, ожидается целое число." << std::endl;
        return 1;
    }

    // Установка количества потоков для MKL
    mkl_set_num_threads(std::thread::hardware_concurrency());

    std::vector<double> A(n * n);
    std::vector<double> A_inv(n * n);

    generate_positive_definite_matrix(A.data(), n);

    // Обращение матрицы и замер времени
    std::copy(A.begin(), A.end(), A_inv.begin());
    auto start = std::chrono::high_resolution_clock::now();
    
    // LU разложение
    std::vector<lapack_int> ipiv(n);
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Ошибка в LAPACKE_dgetrf: " << info << std::endl;
        return 1;
    }

    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A_inv.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "Ошибка в LAPACKE_dgetri: " << info << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Время, затраченное на обращение матрицы размерности "
              << n << "x" << n << ": "
              << diff.count() << " секунд" << std::endl;

    return 0;
}

