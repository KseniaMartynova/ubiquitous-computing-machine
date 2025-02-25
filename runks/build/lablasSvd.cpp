#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>

std::vector<double> generate_spd_matrix(int n) {
    std::vector<double> mat(n*n);
    std::vector<double> B(n*n);

    #pragma omp parallel
    {
        std::minstd_rand0 gen(omp_get_thread_num() + 42);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        #pragma omp for
        for(int i = 0; i < n*n; ++i)
            B[i] = dis(gen);
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, n, 1.0, B.data(), n, B.data(), n, 0.0, mat.data(), n);
    
    #pragma omp parallel for
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
    int lwork = 3*n + std::max(n, 256);
    std::vector<double> work(lwork);

    // Замер времени SVD
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

    const double max_s = *std::max_element(S.begin(), S.end());
    for(int i = 0; i < n; ++i) {
        if(S[i] < 1e-12 * max_s) {
            std::cerr << "Ill-conditioned matrix (S[" << i << "] = " << S[i] << ")" << std::endl;
            return 1;
        }
    }

    // Замер времени инверсии
    auto inv_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        cblas_dscal(n, 1.0/S[i], &U[i*n], 1);
    }

    std::vector<double> A_inv(n*n);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
                n, n, n, 1.0, 
                VT.data(), n,
                U.data(), n,
                0.0, A_inv.data(), n);
    
    auto inv_end = std::chrono::high_resolution_clock::now();

    // Проверка точности
    std::vector<double> identity(n*n, 0.0);
    for(int i = 0; i < n; ++i) identity[i + i*n] = 1.0;
    
    std::vector<double> residual(n*n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0,
                A.data(), n,
                A_inv.data(), n,
                -1.0, residual.data(), n);
    
    double error = cblas_dnrm2(n*n, residual.data(), 1);
    
    // Вывод времени
    std::chrono::duration<double> svd_duration = svd_end - svd_start;
    std::chrono::duration<double> inv_duration = inv_end - inv_start;
    
    std::cout << "Matrix size: " << n << "x" << n
              << "\nSVD time: " << svd_duration.count() << " s"
              << "\nInversion time: " << inv_duration.count() << " s"
              << "\nTotal time: " << (svd_duration + inv_duration).count() << " s"
              << "\nInversion error: " << error
              << "\nCondition number: " << max_s/S.back() << std::endl;

    return 0;
}
