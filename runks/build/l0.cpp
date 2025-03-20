#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <lapacke.h>

// Функция для генерации симметричной положительно определенной матрицы
std::vector<double> create_spd_matrix(int n) {
    std::vector<double> B(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Генерируем случайную матрицу B
    for (auto& val : B) val = dis(gen);

    // Вычисляем A = B * B^T + n*I (column-major)
    std::vector<double> A(n * n, 0.0);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
                n, n, n, 1.0, B.data(), n, B.data(), n, 0.0, A.data(), n);
    
    for (int i = 0; i < n; ++i) 
        A[i + i * n] += n;

    return A;
}

// Функция для вывода матрицы
void print_matrix(const std::vector<double>& matrix, int n, const std::string& name) {
    std::cout << "Матрица " << name << ":\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i + j * n] << " ";  // Column-major порядок
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    const int n = std::stoi(argv[1]);

    // Генерация исходной матрицы
    std::vector<double> A = create_spd_matrix(n);
    print_matrix(A, n, "A (исходная)");

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

    // Вывод сингулярных значений
    std::cout << "Сингулярные значения:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "sigma[" << i << "] = " << S[i] << "\n";
    }

    // Вывод матрицы U
    print_matrix(U, n, "U (левые сингулярные векторы)");

    // Вывод матрицы S (диагональная)
    std::vector<double> S_diag(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        S_diag[i + i * n] = S[i];
    }
    print_matrix(S_diag, n, "S (сингулярные значения)");

    // Вывод матрицы VT (V транспонированная)
    print_matrix(VT, n, "VT (правые сингулярные векторы)");

    // Вычисляем число обусловленности
    double sigma_max = S[0];  // Максимальное сингулярное значение
    double sigma_min = S[n - 1];  // Минимальное сингулярное значение
    double cond_number = sigma_max / sigma_min;

    std::cout << "Число обусловленности: " << cond_number << "\n";

    // Проверка минимального сингулярного значения
    if (sigma_min < 1e-10) {
        std::cerr << "Предупреждение: минимальное сингулярное значение близко к нулю (" 
                  << sigma_min << "). Матрица может быть вырожденной.\n";
    }

    // Проверка SVD: A ≈ U * S * VT
    std::vector<double> A_reconstructed(n * n);
    std::vector<double> US(n * n);

    // Вычисляем U * S
    cblas_dgemm(
        CblasColMajor, 
        CblasNoTrans, 
        CblasNoTrans, 
        n, n, n, 
        1.0, 
        U.data(), n, 
        S_diag.data(), n, 
        0.0, 
        US.data(), n
    );

    // Вычисляем U * S * VT
    cblas_dgemm(
        CblasColMajor, 
        CblasNoTrans, 
        CblasNoTrans, 
        n, n, n, 
        1.0, 
        US.data(), n, 
        VT.data(), n, 
        0.0, 
        A_reconstructed.data(), n
    );

    // Вывод матрицы после SVD
    print_matrix(A_reconstructed, n, "A_reconstructed (U * S * VT)");

    // Вычисляем разность A - U * S * VT
    std::vector<double> svd_residual(n * n);
    for (int i = 0; i < n * n; ++i) {
        svd_residual[i] = A[i] - A_reconstructed[i];
    }
    print_matrix(svd_residual, n, "A - U * S * VT (разность)");

    // Масштабируем строки VT (столбцы V) на 1/S[i]
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

    // Вывод матрицы после инверсии
    print_matrix(A_inv, n, "A_inv (обратная)");

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

    for (int i = 0; i < n; ++i) 
        residual[i + i * n] += 1.0;

    // Вывод разности A * A_inv - I
    print_matrix(residual, n, "A * A_inv - I (разность)");

    // Вывод времени выполнения
    auto svd_duration = std::chrono::duration<double>(svd_end - svd_start);
    auto inv_duration = std::chrono::duration<double>(inv_end - inv_start);
    auto total_duration = svd_duration + inv_duration;

    std::cout << "Время выполнения SVD: " << svd_duration.count() << " s\n"
              << "Время выполнения инверсии: " << inv_duration.count() << " s\n"
              << "Общее время: " << total_duration.count() << " s\n";

    return 0;
}
