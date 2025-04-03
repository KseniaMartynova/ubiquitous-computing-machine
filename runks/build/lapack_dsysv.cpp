#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>     // OpenBLAS
#include <lapacke.h>   // LAPACK (входит в OpenBLAS)

// Генерация случайной симметричной матрицы
void generateSymmetricMatrix(std::vector<double>& A, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Заполняем верхний треугольник
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            A[i * n + j] = dist(gen);
        }
    }
    
    // Зеркально копируем в нижний треугольник
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i * n + j] = A[j * n + i];
        }
    }
}

// Обращение симметричной матрицы через DSYSV (AX = I)
double invertSymmetricMatrix(std::vector<double>& A, std::vector<double>& Ainv, int n) {
    Ainv = A; // Копируем исходную матрицу
    
    // Создаем единичную матрицу (правая часть AX = I)
    std::vector<double> I(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        I[i * n + i] = 1.0;
    }

    // Параметры для LAPACK
    char uplo = 'U';  // Используем верхний треугольник
    lapack_int nrhs = n;     // Количество правых частей
    lapack_int lda = n;      // Ведущая размерность A
    lapack_int ldb = n;      // Ведущая размерность B
    std::vector<lapack_int> ipiv(n); // Вектор перестановок

    // Рабочий массив (запрашиваем оптимальный размер)
    lapack_int lwork = -1;
    double lwork_query;
    lapack_int info = LAPACKE_dsysv_work(
        LAPACK_ROW_MAJOR, uplo, n, nrhs, 
        Ainv.data(), lda, ipiv.data(), 
        I.data(), ldb, &lwork_query, lwork
    );

    lwork = (lapack_int)lwork_query;
    std::vector<double> work(lwork);

    // Замер времени и вызов DSYSV
    auto start = std::chrono::high_resolution_clock::now();
    info = LAPACKE_dsysv_work(
        LAPACK_ROW_MAJOR, uplo, n, nrhs, 
        Ainv.data(), lda, ipiv.data(), 
        I.data(), ldb, work.data(), lwork
    );
    auto end = std::chrono::high_resolution_clock::now();

    if (info != 0) {
        std::cerr << "DSYSV failed with info = " << info << std::endl;
        return -1.0;
    }

    Ainv = I; // Результат (обратная матрица)
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    const int n = 512; // Размер матрицы
    std::vector<double> A(n * n);
    std::vector<double> Ainv(n * n);

    // Генерация симметричной матрицы
    generateSymmetricMatrix(A, n);

    // Обращение матрицы и замер времени
    double time = invertSymmetricMatrix(A, Ainv, n);

    if (time >= 0) {
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        std::cout << "Inversion time: " << time << " seconds" << std::endl;
    }

    return 0;
}
