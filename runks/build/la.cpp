#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>

std::vector<double> generate_spd_matrix(int n) {
    std::vector<double> mat(n*n);
    std::vector<double> B(n*n);

    std::minstd_rand0 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for(int i = 0; i < n*n; ++i)
        B[i] = dis(gen);

    // A = B * B^T
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, n, 1.0, B.data(), n, B.data(), n, 0.0, mat.data(), n);
    
    // Добавляем n*I для улучшения обусловленности
    for(int i = 0; i < n; ++i)
        mat[i + i*n] += n;

    return mat;
}

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }
    
    const int n = std::stoi(argv[1]);
    auto A = generate_spd_matrix(n);
    
    std::vector<double> S(n), U(n*n), VT(n*n);

    // Запрос оптимального размера рабочего массива
    double query_work;
    int lwork_query = -1;
    int info_query = LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, 'A', 'A',
                                        n, n, A.data(), n, S.data(),
                                        U.data(), n, VT.data(), n,
                                        &query_work, lwork_query);

    if (info_query != 0) {
        std::cerr << "Failed to query optimal workspace size: " << info_query << std::endl;
        return 1;
    }

    int lwork = static_cast<int>(query_work);
    std::vector<double> work(lwork);

    // Вызов SVD
    auto svd_start = std::chrono::high_resolution_clock::now();
    int info = LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, 'A', 'A',
                                  n, n, A.data(), n, S.data(),
                                  U.data(), n, VT.data(), n,
                                  work.data(), lwork);
    auto svd_end = std::chrono::high_resolution_clock::now();
    
    if(info != 0) {
        std::cerr << "SVD failed with code: " << info << std::endl;
        return 1;
    }

    // Проверка сингулярных чисел
    const double max_s = *std::max_element(S.begin(), S.end());
    const double min_s = *std::min_element(S.begin(), S.end());
    if(min_s < 1e-12 * max_s) {
        std::cerr << "Ill-conditioned matrix (min_s = " << min_s << ")" << std::endl;
        return 1;
    }

    // Инверсия через SVD (исправленная версия)
    auto inv_start = std::chrono::high_resolution_clock::now();
    
    // Масштабируем ПРАВЫЕ сингулярные векторы (V) = столбцы VT^T
    for(int i = 0; i < n; ++i) {
        cblas_dscal(n, 1.0/S[i], &VT[i], n); // Шаг между элементами вектора = n
    }

    // A_inv = V * S^{-1} * U^T = VT^T * U^T
    std::vector<double> A_inv(n*n);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
                n, n, n, 1.0, 
                VT.data(), n,  // VT^T = V
                U.data(), n,   // U^T
                0.0, A_inv.data(), n);
    
    auto inv_end = std::chrono::high_resolution_clock::now();

    // Проверка точности: ||A * A_inv - I||_F
    std::vector<double> residual(n*n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0,
                A.data(), n,
                A_inv.data(), n,
                -1.0, residual.data(), n);
    
    // Добавляем единичную матрицу к residual
    for(int i = 0; i < n; ++i)
        residual[i + i*n] += 1.0;
    
    double error = cblas_dnrm2(n*n, residual.data(), 1);

    // Вывод результатов
    std::chrono::duration<double> svd_duration = svd_end - svd_start;
    std::chrono::duration<double> inv_duration = inv_end - inv_start;
    
    std::cout << "Matrix size: " << n << "x" << n
              << "\nSVD time: " << svd_duration.count() << " s"
              << "\nInversion time: " << inv_duration.count() << " s"
              << "\nTotal time: " << (svd_duration + inv_duration).count() << " s"
              << "\nInversion error: " << error
              << "\nCondition number: " << max_s/min_s << std::endl;

    return 0;
}
