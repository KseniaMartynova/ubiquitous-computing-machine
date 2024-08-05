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

int main() {
     int n;
     std::cout << "Введите размерность матрицы: ";
     std::cin >> n;

     std::vector<double> A(n * n);
     std::vector<double> A_inv(n * n);

     generate_positive_definite_matrix(A.data(), n);

     // Обращение матрицы и замер времени
     std::copy(A.begin(), A.end(), A_inv.begin());
     auto start = std::chrono::high_resolution_clock::now();

     std::vector<lapack_int> ipiv(n);
     int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_inv.data(), n,
ipiv.data());
     info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A_inv.data(), n,
ipiv.data());

     auto end = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> diff = end - start;

     std::cout << "Время, затраченное на обращение матрицы размерности "
<< n << "x" << n << ": "
               << diff.count() << " секунд" << std::endl;

     return 0;
}
