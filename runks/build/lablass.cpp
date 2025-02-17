#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>

std::vector<double> generate_spd_matrix(int n) {
    std::vector<double> mat(n*n);
    std::vector<double> B(n*n);
    
    // Параллельная генерация случайной матрицы
    #pragma omp parallel
    {
        std::minstd_rand0 gen(omp_get_thread_num() + 42);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        #pragma omp for
        for(int i = 0; i < n*n; ++i) 
            B[i] = dis(gen);
    }

    // Создание положительно определенной матрицы: A = B*B^T + n*I
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                n, n, n, 1.0, B.data(), n, B.data(), n, 0.0, mat.data(), n);
    
    #pragma omp parallel for
    for(int i = 0; i < n; ++i)
        mat[i*n + i] += n;
    
    return mat;
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }
    
    const int n = std::stoi(argv[1]);
    auto A = generate_spd_matrix(n);
    
    // Выделение памяти для SVD
    std::vector<double> U(n*n), VT(n*n), S(n);
    std::vector<double> work(3*n);
    
    // Настройка многопоточности
    openblas_set_num_threads(omp_get_max_threads());
    
    // Замер времени SVD
    auto start = std::chrono::high_resolution_clock::now();
    
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', 
                             n, n, A.data(), n, S.data(),
                             U.data(), n, VT.data(), n, work.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    if(info != 0) {
        std::cerr << "SVD failed with code: " << info << std::endl;
        return 1;
    }
    
    // Вывод результатов
    std::cout << "Matrix: " << n << "x" << n
              << "\nThreads: " << openblas_get_num_threads()
              << "\nSVD Time: " << elapsed.count() << " s"
              << "\nMax σ: " << S[0]
              << "\nMin σ: " << S[n-1] << std::endl;
    
    return 0;
}
