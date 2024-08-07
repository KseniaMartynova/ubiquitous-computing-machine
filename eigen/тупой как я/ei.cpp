#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    int N;
    std::cout << "Enter the size of the matrix: ";
    std::cin >> N;

    if (N <= 0) {
        std::cerr << "Matrix size must be a positive integer." << std::endl;
        return 1;
    }

    // Создание положительно определенной матрицы
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    A = A * A.transpose(); // Симметричная матрица
    A = A + Eigen::MatrixXd::Identity(N, N) * N; // Положительно определенная матрица

    // Обращение матрицы и замер времени выполнения
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd A_inv = A.inverse();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Вывод времени выполнения
    std::cout << "Time to invert matrix: " << duration.count() << " seconds" << std::endl;

    // Проверка корректности обращения
    Eigen::MatrixXd I = A * A_inv;
    Eigen::MatrixXd I_expected = Eigen::MatrixXd::Identity(N, N);
    double error = (I - I_expected).norm();

    // Вывод результата проверки
    if (error < 1e-10) {
        std::cout << "Matrix inversion is correct." << std::endl;
    } else {
        std::cout << "Matrix inversion is incorrect." << std::endl;
    }

    return 0;
}
