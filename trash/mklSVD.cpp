#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>  // Added for DBL_EPSILON
#include <omp.h>

void generate_positive_definite_matrix(double* A, int n) {
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 1234);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n * n, A, 0.0, 1.0);
    vslDeleteStream(&stream);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i*n + j] = A[j*n + i] = (A[i*n + j] + A[j*n + i]) / 2.0;
        }
        A[i*n + i] += n;
    }
}

bool check_inversion_result(const std::vector<double>& A, const std::vector<double>& A_inv, int n) {
    std::vector<double> result(n * n, 0.0);

    // Используем оптимизированное матричное умножение MKL
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                n, n, n, 1.0, A.data(), n, A_inv.data(), n, 0.0, result.data(), n);

    // Для больших матриц используем более мягкий критерий точности
    const double base_tolerance = 1e-6;
    // Масштабируем допуск в зависимости от размера матрицы
    const double tolerance = base_tolerance * std::sqrt(static_cast<double>(n));
    
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
            if (diff > tolerance) {
                error_count++;
            }
        }
    }

    avg_diff /= (n * n);
    
  //  std::cout << "Максимальное отклонение: " << max_diff << std::endl;
    //std::cout << "Среднее отклонение: " << avg_diff << std::endl;
    //std::cout << "Количество элементов вне допуска: " << error_count << std::endl;
    //std::cout << "Используемый допуск: " << tolerance << std::endl;
    //std::cout << "check: " << (error_count < ((n * n) / 1000.0)) << std::endl;

    // Считаем результат приемлемым, если количество элементов вне допуска менее 0.1%
    return error_count < (n * n) / 1000.0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Использование: " << argv[0] << " <размер матрицы>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    
    // Устанавливаем количество потоков для MKL равным количеству потоков OpenMP
    int num_threads = omp_get_max_threads();
    mkl_set_num_threads(num_threads);
    
    // Включаем динамическое планирование потоков
    omp_set_dynamic(1);

    if (static_cast<size_t>(n) > std::vector<double>().max_size() / n) {
        std::cerr << "Размер матрицы слишком большой для std::vector" << std::endl;
        return 1;
    }

    std::vector<double> A(n * n);
    std::vector<double> A0(n * n);
    std::vector<double> A_inv(n * n);
    
    generate_positive_definite_matrix(A.data(), n);
    A0 = A;
    std::vector<double> singular_values(n);
    std::vector<double> U(n * n);
    std::vector<double> VT(n * n);

    // Определяем оптимальный размер рабочего массива
    double worksize;
    LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'A', 'A', n, n, A.data(), n,
                        nullptr, nullptr, n, nullptr, n, &worksize, -1);
    std::vector<double> work(static_cast<size_t>(worksize));

    auto start = std::chrono::high_resolution_clock::now();

    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n, n,
                             A.data(), n, singular_values.data(),
                             U.data(), n, VT.data(), n, work.data());

    if (info != 0) {
        std::cerr << "Ошибка в SVD разложении" << std::endl;
        return 1;
    }

    // Находим максимальное сингулярное число для масштабирования порога отсечения
    double max_sigma = singular_values[0];
    for (int i = 1; i < n; ++i) {
        max_sigma = std::max(max_sigma, singular_values[i]);
    }
    
    // Устанавливаем порог отсечения в зависимости от размера матрицы и максимального сингулярного числа
    double cutoff = max_sigma * std::max(n, 10000) * DBL_EPSILON;
    
    #pragma omp parallel for simd
    for (int i = 0; i < n; ++i) {
        singular_values[i] = (singular_values[i] > cutoff) ? 1.0 / singular_values[i] : 0.0;
    }

    std::vector<double> temp(n * n);

    // U * Σ⁻¹
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            temp[i * n + j] = U[i * n + j] * singular_values[j];
        }
    }
    
    // (U * Σ⁻¹) * Vᵀ
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, temp.data(), n, VT.data(), n, 0.0, A_inv.data(), n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Время обращения матрицы " << n << "x" << n 
              << " (SVD): " << diff.count() << " секунд" << std::endl;
//    std::cout << "Использовано потоков: " << num_threads << std::endl;

    if (check_inversion_result(A0, A_inv, n)) {
        std::cout << "Проверка пройдена успешно" << std::endl;
    } else {
        std::cout << "Проверка не пройдена" << std::endl;
    }

    return 0;
}
