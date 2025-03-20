#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <algorithm>

extern "C" {
    void dgesdd_(char* jobz, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu,
                double* vt, int* ldvt, double* work,
                int* lwork, int* iwork, int* info);
}

using namespace std;
using namespace chrono;

double matrix_norm(const vector<double>& mat, int n) {
    double max = 0.0;
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += fabs(mat[i*n + j]);
        }
        if (sum > max) max = sum;
    }
    return max;
}

void generate_spd_matrix(vector<double>& A, int n) {
    const double diag_boost = n * 1e-3; // Гарантия положительной определенности
    
    // Генерация случайной симметричной матрицы
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            A[i*n + j] = static_cast<double>(rand()) / RAND_MAX;
            if (i != j) A[j*n + i] = A[i*n + j];
        }
        A[i*n + i] += diag_boost; // Усиление диагонали
    }
}

void svd_inversion(vector<double>& A, int n) {
    char jobz = 'A';
    int m = n, lda = n, info;
    vector<double> s(n), u(n*n), vt(n*n);
    vector<double> A_work = A;

    // Определение оптимального рабочего пространства
    int lwork = -1;
    double work_query;
    dgesdd_(&jobz, &m, &n, A_work.data(), &lda, s.data(),
            u.data(), &n, vt.data(), &n, &work_query, &lwork, nullptr, &info);
    
    lwork = static_cast<int>(work_query);
    vector<double> work(lwork);
    vector<int> iwork(8*n);

    // Выполнение SVD
    dgesdd_(&jobz, &m, &n, A_work.data(), &lda, s.data(),
            u.data(), &n, vt.data(), &n, work.data(), &lwork, iwork.data(), &info);
    
    if (info != 0) throw runtime_error("SVD failed: " + to_string(info));

    // Инвертирование сингулярных значений с проверкой
    double eps = 1e-12 * *max_element(s.begin(), s.end());
    for (auto& val : s) {
        if (val < eps) throw runtime_error("Singular matrix detected");
        val = 1.0 / val;
    }

    // Вычисление A^{-1} = V * S^{-1} * U^T
    vector<double> inv(n*n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += vt[k*n + i] * s[k] * u[j*n + k];
            }
            inv[i*n + j] = sum;
        }
    }
    A.swap(inv);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;
    }

    const int n = stoi(argv[1]);
    srand(42);

    vector<double> A(n*n);
    generate_spd_matrix(A, n);
    const auto A_orig = A;

    try {
        auto start = high_resolution_clock::now();
        svd_inversion(A, n);
        auto duration = duration_cast<duration<double>>(high_resolution_clock::now() - start);
        
        // Проверка ошибок
        double max_err = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += A_orig[i*n + k] * A[k*n + j];
                }
                double target = (i == j) ? 1.0 : 0.0;
                max_err = max(max_err, fabs(sum - target));
            }
        }

        cout << fixed << setprecision(3);
        cout << "Inversion time: " << duration.count() << " s\n";
        cout << scientific << "Max error: " << max_err << endl;
        cout << "Relative error: " << max_err / matrix_norm(A_orig, n) << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}
