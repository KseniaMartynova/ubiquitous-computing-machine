#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);

    if (N <= 0) {
        std::cerr << "Matrix size must be a positive integer." << std::endl;
        return 1;
    }

    // Создание двух положительно определенных матриц
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    A = A * A.transpose(); // Симметричная матрица
    A = A + Eigen::MatrixXd::Identity(N, N) * N; // Положительно определенная матрица

    Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
    B = B * B.transpose(); // Симметричная матрица
    B = B + Eigen::MatrixXd::Identity(N, N) * N; // Положительно определенная матрица

    // Умножение матриц и замер времени выполнения
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd C = A * B;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Вывод времени выполнения
    std::cout << "Time to multiply matrices: " << duration.count() << " seconds" << std::endl;

    return 0;
}
