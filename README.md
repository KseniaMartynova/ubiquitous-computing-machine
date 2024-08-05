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
1. В том же контейнере intel запускаем openblas.cpp c помощью: icpx -qmkl openblas.cpp -o oopn  и ./oopn

## LAPACK
1. В том же контейнере intel запускаем lapack.cpp с помощью: icpx -qmkl lapack.cpp -o lapa  и ./lapa

## Eigen
Я все запихала в контейнер, но можно и без него.
1. sudo apt update && sudo apt upgrade
2. sudo apt install libeigen3-dev
3. dpkg -L libeigen3-dev
4. Запускаем файл eigen.cpp с помощью g++ -I /usr/include/eigen3/ eigen.cpp -o ei  и  ./ei

Вот по этой ссылке делала я:  https://www.cyberithub.com/how-to-install-eigen3-on-ubuntu-20-04-lts-focal-fossa/

## Armadillo
Тут я тоже запихала в конейнер, но лучше без него.
1. sudo apt update && sudo apt upgrade
2. sudo apt install cmake libopenblas-dev liblapack-dev
3. Скачать отсюда: https://arma.sourceforge.net/download.html    само Armadillo
4. Потом найти то, что скачали и разархивировать и перейти в эту разахивированную директорию:  cd arma*
5. cmake .
6. make
7. sudo make install
8. запускаем apm.cpp:  g++ arm.cpp -o arm -DARMA_DONT_USE_WRAPPER -lopenblas -llapack  и ./arm

Ссылка, по которой делала я: https://solarianprogrammer.com/2017/03/24/getting-started-armadillo-cpp-linear-algebra-windows-mac-linux/

## NumPy
1. Запускаем файл NumPy.py:  python3 NumPy.py

