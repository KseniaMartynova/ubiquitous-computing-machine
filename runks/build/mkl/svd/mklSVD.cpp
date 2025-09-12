#include <iostream>
#include <vector>
#include <chrono>
#include <mkl.h>
#include <algorithm>  // Добавлено
#include <cfloat> 
#include <omp.h>

// Специализированная функция для обращения матрицы через SVD
void invert_matrix_via_svd(double* A, int n) {
    // Выделение памяти для SVD
    std::vector<double> U(n * n);
    std::vector<double> VT(n * n);
    std::vector<double> S(n);

    // Выполнение SVD
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', n, n, A, n, S.data(), U.data(), n, VT.data(), n);
    if (info != 0) {
        std::cerr << "Ошибка в SVD разложении: " << info << std::endl;
        return;
    }

    // Инверсия сингулярных значений
    double max_s = *std::max_element(S.begin(), S.end());
    double threshold = max_s * n * DBL_EPSILON;  // Порог для отсечения малых значений

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        S[i] = (S[i] > threshold) ? 1.0 / S[i] : 0.0;
    }

    // Вычисление V * Σ^{-1}
    std::vector<double> VSigma_inv(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            VSigma_inv[i * n + j] = VT[j * n + i] * S[j];  // VT хранится как V^T, поэтому транспонируем
        }
    }

    // Вычисление A^{-1} = VΣ^{-1} * U^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                n, n, n, 1.0, VSigma_inv.data(), n, U.data(), n, 0.0, A, n);
}

// Генерация положительно определенной матрицы
void generate_positive_definite_matrix(double* A, int n) {
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 1234);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n * n, A, 0.0, 1.0);
    vslDeleteStream(&stream);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i * n + j] = A[j * n + i] = (A[i * n + j] + A[j * n + i]) / 2.0;
        }
        A[i * n + i] += n;  // Делаем диагональные элементы доминирующими
    }
}

// Проверка результата обращения
bool check_inversion_result(const double* A, const double* A_inv, int n) {
    std::vector<double> result(n * n, 0.0);

    // Умножение A на A_inv
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, 1.0, A, n, A_inv, n, 0.0, result.data(), n);

    // Проверка отклонения от единичной матрицы
    double max_diff = 0.0;
    double avg_diff = 0.0;
    int error_count = 0;

    #pragma omp parallel for reduction(max:max_diff) reduction(+:avg_diff,error_count)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = std::abs(result[i * n + j] - expected);
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            if (diff > 1e-6) {  // Порог для ошибки
                error_count++;
            }
        }
    }

    avg_diff /= (n * n);

    std::cout << "Максимальное отклонение: " << max_diff << std::endl;
    std::cout << "Среднее отклонение: " << avg_diff << std::endl;
    std::cout << "Количество элементов вне допуска: " << error_count << std::endl;

    // Результат считается успешным, если менее 0.1% элементов вне допуска
    return error_count < (n * n) / 1000;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Использование: " << argv[0] << " <размер матрицы>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Размер матрицы должен быть положительным числом" << std::endl;
        return 1;
    }

    // Настройка многопоточности
    int num_threads = omp_get_max_threads();
    mkl_set_num_threads(num_threads);
    omp_set_dynamic(1);

    // Выделение памяти
    std::vector<double> A(n * n);
    std::vector<double> A0(n * n);

    // Генерация матрицы
    generate_positive_definite_matrix(A.data(), n);
    A0 = A;  // Сохраняем оригинальную матрицу

    // Замер времени
    auto start = std::chrono::high_resolution_clock::now();

    // Обращение матрицы через SVD
    invert_matrix_via_svd(A.data(), n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Время обращения матрицы " << n << "x" << n 
              << " (SVD): " << diff.count() << " секунд" << std::endl;
    std::cout << "Использовано потоков: " << num_threads << std::endl;

    // Проверка результата
    if (check_inversion_result(A0.data(), A.data(), n)) {
        std::cout << "Проверка пройдена успешно" << std::endl;
    } else {
        std::cout << "Проверка не пройдена" << std::endl;
    }

    return 0;
}
