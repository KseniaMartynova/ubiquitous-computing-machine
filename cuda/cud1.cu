#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <chrono>
#include <cmath>
#include <cstdlib>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSOLVER(func)                                                   \
{                                                                              \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("cuSOLVER API failed at line %d with error: %d\n",              \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("cuBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return EXIT_FAILURE;
    }

    int N = std::atoi(argv[1]);
    const int lda = N;  // Линейный размер матрицы

    // Выделение памяти на хосте
    double *h_A = (double*)malloc(lda * N * sizeof(double));
    double *h_A_inv = (double*)malloc(lda * N * sizeof(double));
    double *h_I = (double*)malloc(lda * N * sizeof(double));

    // Заполнение матрицы случайными значениями
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * lda + j] = (double)rand() / RAND_MAX;
        }
    }

    // Создание положительно определённой матрицы
    double *h_A_posdef = (double*)malloc(lda * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A_posdef[i * lda + j] = h_A[i * lda + j];
        }
    }
    for (int i = 0; i < N; i++) {
        h_A_posdef[i * lda + i] += N;
    }

    // Выделение памяти на устройстве
    double *d_A;
    CHECK_CUDA(cudaMalloc((void**)&d_A, lda * N * sizeof(double)));

    // Копирование матрицы на устройство
    CHECK_CUDA(cudaMemcpy(d_A, h_A_posdef, lda * N * sizeof(double), cudaMemcpyHostToDevice));

    // Инициализация cuSOLVER
    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    // Выделение памяти для работы cuSOLVER
    int *devInfo;
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));

    double *d_work;
    int lwork;
    CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_UPPER, N, d_A, lda, &lwork));
    CHECK_CUDA(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

    // Измерение времени выполнения
    auto start = std::chrono::high_resolution_clock::now();

    // Выполнение разложения Холецкого
    CHECK_CUSOLVER(cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_UPPER, N, d_A, lda, d_work, lwork, devInfo));

    // Выполнение обращения матрицы
    CHECK_CUSOLVER(cusolverDnDpotri(cusolverH, CUBLAS_FILL_MODE_UPPER, N, d_A, lda, d_work, lwork, devInfo));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Копирование результата на хост
    CHECK_CUDA(cudaMemcpy(h_A_inv, d_A, lda * N * sizeof(double), cudaMemcpyDeviceToHost));

    // Проверка корректности обращения
    double *d_I;
    CHECK_CUDA(cudaMalloc((void**)&d_I, lda * N * sizeof(double)));

    // Инициализация cuBLAS
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));

    // Умножение исходной матрицы на обратную
    double alpha = 1.0;
    double beta = 0.0;
    CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, lda, d_A, lda, &beta, d_I, lda));

    // Копирование результата на хост
    CHECK_CUDA(cudaMemcpy(h_I, d_I, lda * N * sizeof(double), cudaMemcpyDeviceToHost));

    // Проверка на единичную матрицу
    bool correct = true;
    double tolerance = 1e-6;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(h_I[i * lda + j] - expected) > tolerance) {
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    if (correct) {
        std::cout << "Matrix inversion is correct." << std::endl;
    } else {
        std::cout << "Matrix inversion is incorrect." << std::endl;
    }

    // Вывод времени выполнения
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    // Освобождение памяти
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_I));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    CHECK_CUBLAS(cublasDestroy(cublasH));

    free(h_A);
    free(h_A_inv);
    free(h_A_posdef);
    free(h_I);

    return 0;
}
