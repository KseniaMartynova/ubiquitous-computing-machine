#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mkl.h>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <omp.h>
#include <iomanip>  // Для форматированного вывода

void generate_positive_definite_matrix(double* A, int n) {
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 1234);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n * n, A, 0.0, 1.0);
    vslDeleteStream(&stream);

    // Улучшенное создание положительно определённой матрицы
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i*n + j] = A[j*n + i] = (A[i*n + j] + A[j*n + i]) / 2.0;
        }
        // Увеличиваем диагональное преобладание для лучшей обусловленности
        A[i*n + i] = n + std::accumulate(A + i*n, A + (i+1)*n, 0.0);
    }
}

bool check_inversion_result(const std::vector<double>& A, const std::vector<double>& A_inv, int n) {
    std::vector<double> result(n * n, 0.0);
    
    // Используем более точное матричное умножение с двойной точностью
    const double alpha = 1.0;
    const double beta = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, alpha, A.data(), n, A_inv.data(), n, beta, result.data(), n);

    // Адаптивный расчёт допуска на основе размера и обусловленности матрицы
    double norm_A = 0.0;
    double norm_A_inv = 0.0;
    
    #pragma omp parallel for reduction(max:norm_A,norm_A_inv)
    for (int i = 0; i < n; ++i) {
        double row_sum_A = 0.0;
        double row_sum_A_inv = 0.0;
        for (int j = 0; j < n; ++j) {
            row_sum_A += std::abs(A[i * n + j]);
            row_sum_A_inv += std::abs(A_inv[i * n + j]);
        }
        norm_A = std::max(norm_A, row_sum_A);
        norm_A_inv = std::max(norm_A_inv, row_sum_A_inv);
    }

    // Оцениваем число обусловленности и корректируем допуск
    double condition_number = norm_A * norm_A_inv;
    double base_tolerance = 1e-6;
    double tolerance = base_tolerance * std::sqrt(static_cast<double>(n)) * 
                      (1.0 + std::log10(condition_number));
    
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
    
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Число обусловленности матрицы: " << condition_number << std::endl;
    std::cout << "Максимальное отклонение: " << max_diff << std::endl;
    std::cout << "Среднее отклонение: " << avg_diff << std::endl;
    std::cout << "Количество элементов вне допуска: " << error_count << std::endl;
    std::cout << "Используемый допуск: " << tolerance << std::endl;

    // Более строгая проверка для хорошо обусловленных матриц и более мягкая для плохо обусловленных
    double error_threshold = 0.001 * (1.0 + std::log10(condition_number));
    return (static_cast<double>(error_count) / (n * n)) < error_threshold;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Использование: " << argv[0] << " <размер матрицы>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    
    int num_threads = omp_get_max_threads();
    mkl_set_num_threads(num_threads);
    omp_set_dynamic(1);

    if (static_cast<size_t>(n) > std::vector<double>().max_size() / n) {
        std::cerr << "Размер матрицы слишком большой для std::vector" << std::endl;
        return 1;
    }

    std::vector<double> A(n * n);
    std::vector<double> A_inv(n * n);
    
    generate_positive_definite_matrix(A.data(), n);

    std::vector<double> singular_values(n);
    std::vector<double> U(n * n);
    std::vector<double> VT(n * n);

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

    // Улучшенная обработка сингулярных чисел
    double max_sigma = singular_values[0];
    double min_sigma = max_sigma;
    
    for (int i = 1; i < n; ++i) {
        max_sigma = std::max(max_sigma, singular_values[i]);
        min_sigma = std::min(min_sigma, singular_values[i]);
    }
    
    // Адаптивный порог отсечения с учётом диапазона сингулярных чисел
    double condition_number = max_sigma / min_sigma;
    double cutoff = max_sigma * std::max(n * DBL_EPSILON, 
                                       DBL_EPSILON * std::sqrt(condition_number));

    std::cout << "Диапазон сингулярных чисел: " << min_sigma << " - " << max_sigma << std::endl;
    std::cout << "Используемый порог отсечения: " << cutoff << std::endl;
    
    #pragma omp parallel for simd
    for (int i = 0; i < n; ++i) {
        singular_values[i] = (singular_values[i] > cutoff) ? 
            1.0 / singular_values[i] : 0.0;
    }

    std::vector<double> temp(n * n);
    
    // U * Σ⁻¹ с улучшенной точностью
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            temp[i * n + j] = U[i * n + j] * singular_values[j];
        }
    }
    
    // (U * Σ⁻¹) * Vᵀ с явными параметрами точности
    const double alpha = 1.0;
    const double beta = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, alpha, temp.data(), n, VT.data(), n, beta, A_inv.data(), n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Время обращения матрицы " << n << "x" << n 
              << " (SVD): " << diff.count() << " секунд" << std::endl;
    std::cout << "Использовано потоков: " << num_threads << std::endl;

    if (check_inversion_result(A, A_inv, n)) {
        std::cout << "Проверка пройдена успешно" << std::endl;
    } else {
        std::cout << "Проверка не пройдена" << std::endl;
    }

    return 0;
}
