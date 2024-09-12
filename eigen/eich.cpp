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

    // Создание положительно определенной матрицы
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    A = A * A.transpose(); // Симметричная матрица
    A = A + Eigen::MatrixXd::Identity(N, N) * N; // Положительно определенная матрица

    // Обращение матрицы с использованием разложения Холецкого и замер времени выполнения
    auto start = std::chrono::high_resolution_clock::now();

    // Выполняем разложение Холецкого
    Eigen::LLT<Eigen::MatrixXd> lltOfA(A);
    if (lltOfA.info() != Eigen::Success) {
        std::cerr << "Разложение Холецкого не удалось." << std::endl;
        return 1;
    }

    // Вычисляем обратную матрицу
    Eigen::MatrixXd A_inv = lltOfA.solve(Eigen::MatrixXd::Identity(N, N));

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
