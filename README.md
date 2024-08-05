# Сравнение производительности библиотек линейной алгебры для обращения положительно определенных матриц
## Содержание
- [MKL](#Intel_Math_Kernel_Library (MKL))
- [OpenBLAS](#OpenBLAS)
- [LAPACK](#LAPACK)
- [Eigen](#Eigen)
- [Armadillo](#Armadillo)
- [NumPy](#NumPy)
- [CUDA cuBLAS и cuSOLVER](#CUDA_cuBLAS_и_cuSOLVER)
  
## Intel Math Kernel Library (MKL)
1. Скачиваем отсюда: https://hub.docker.com/r/intel/oneapi-basekit докер для intel
2. Запускаем файл mkl.cpp с помощью: icpx -qmkl mkl.cpp -o mkl  и  ./mkl
3. Вводим размерность матрицы

## OpenBLAS
1. В том же контейнере intel запускаем openblass.cpp c помощью: icpx -qmkl openblas.cpp -o oopn  и ./oopn
