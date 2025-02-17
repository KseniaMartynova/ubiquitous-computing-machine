#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mkl.h>
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

    // Параллельное создание положительно определенной матрицы
    #pragma omp parallel
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                    n, n, n, 1.0, B.data(), n, B.data(), n, 0.0, mat.data(), n);
        
        #pragma omp for
        for(int i = 0; i < n; ++i)
            mat[i*n + i] += n;
    }
    
    return mat;
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }
    
    const int n = std::stoi(argv[1]);
    auto A = generate_spd_matrix(n);
    
    // Настройка многопоточности MKL
    mkl_set_num_threads(omp_get_max_threads());
    mkl_set_dynamic(0);
    
    // Выделение памяти для SVD
    std::vector<double> U(n*n), VT(n*n), S(n);
    
    // Замер времени SVD
    auto start = std::chrono::high_resolution_clock::now();
    
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', n, n, 
                             A.data(), n, S.data(),
                             U.data(), n, VT.data(), n);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    if(info != 0) {
        std::cerr << "SVD failed with code: " << info << std::endl;
        return 1;
    }
    
    // Вывод результатов
    std::cout << "Matrix: " << n << "x" << n
          
              << "\nSVD Time: " << elapsed.count() << " s"
    
    return 0;
}
