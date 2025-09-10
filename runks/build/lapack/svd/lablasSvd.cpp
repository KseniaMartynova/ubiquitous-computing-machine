#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>

// Функция для генерации симметричной положительно определенной матрицы
std::vector<double> create_spd_matrix(int n) {
    std::vector<double> B(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Генерируем случайную матрицу B с параллелизацией
    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) 
        B[i] = dis(gen);

    // Вычисляем A = B * B^T + n*I (column-major)
    std::vector<double> A(n * n, 0.0);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
                n, n, n, 1.0, B.data(), n, B.data(), n, 0.0, A.data(), n);
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) 
        A[i + i * n] += n;

    return A;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    const int n = std::stoi(argv[1]);

    // Установка количества потоков для BLAS (если поддерживается)
    // char* env_var = "OMP_NUM_THREADS";
    // setenv(env_var, "4", 1); // Пример для 4 потоков
    // Для MKL можно использовать mkl_set_num_threads()
    // Для OpenBLAS можно использовать openblas_set_num_threads()

    // Генерация исходной матрицы
    std::vector<double> A = create_spd_matrix(n);

    // SVD параметры
    std::vector<double> S(n), U(n * n), VT(n * n);
    int lwork = -1;
    double lwork_query;

    // Запрос оптимального размера рабочего массива
    int info_query = LAPACKE_dgesvd_work(
        LAPACK_COL_MAJOR, 
        'A', 
        'A', 
        n, n, 
        A.data(), n, 
        S.data(), 
        U.data(), n, 
        VT.data(), n, 
        &lwork_query, 
        lwork
    );

    if (info_query != 0) {
        std::cerr << "LWORK query failed: " << info_query << std::endl;
        return 1;
    }

    lwork = static_cast<int>(lwork_query);
    std::vector<double> work(lwork);

    // Выполняем SVD
    auto svd_start = std::chrono::high_resolution_clock::now();
    int info = LAPACKE_dgesvd_work(
        LAPACK_COL_MAJOR, 
        'A', 'A', 
        n, n, 
        A.data(), n, 
        S.data(), 
        U.data(), n, 
        VT.data(), n, 
        work.data(), 
        lwork
    );
    auto svd_end = std::chrono::high_resolution_clock::now();

    if (info != 0) {
        std::cerr << "SVD failed: " << info << std::endl;
        return 1;
    }

    // Масштабируем строки VT (столбцы V) на 1/S[i] с параллелизацией
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) 
        cblas_dscal(n, 1.0 / S[i], &VT[i], n);

    // Вычисляем A_inv = V * S^{-1} * U^T = VT^T * U^T
    auto inv_start = std::chrono::high_resolution_clock::now();
    std::vector<double> A_inv(n * n);
    cblas_dgemm(
        CblasColMajor, 
        CblasTrans,     // VT^T = V
        CblasTrans,     // U^T
        n, n, n, 
        1.0, 
        VT.data(), n,   // V (column-major)
        U.data(), n,    // U^T (column-major)
        0.0, 
        A_inv.data(), n
    );
    auto inv_end = std::chrono::high_resolution_clock::now();

    // Проверка ошибки инверсии: A * A_inv - I
    std::vector<double> residual(n * n);
    cblas_dgemm(
        CblasColMajor, 
        CblasNoTrans, 
        CblasNoTrans, 
        n, n, n, 
        1.0, 
        A.data(), n, 
        A_inv.data(), n, 
        -1.0, 
        residual.data(), n
    );

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) 
        residual[i + i * n] += 1.0;

    // Вычисляем ошибку инверсии
    double error = cblas_dnrm2(n * n, residual.data(), 1);

    // Вывод времени выполнения
    auto svd_duration = std::chrono::duration<double>(svd_end - svd_start);
    auto inv_duration = std::chrono::duration<double>(inv_end - inv_start);
    auto total_duration = svd_duration + inv_duration;

    std::cout << "Время выполнения SVD: " << svd_duration.count() << " s\n"
              << "Время выполнения инверсии: " << inv_duration.count() << " s\n"
              << "Общее время: " << total_duration.count() << " s\n"
              << "Ошибка инверсии: " << error << std::endl;

    return 0;
}
