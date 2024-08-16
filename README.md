# Сравнение производительности библиотек линейной алгебры для обращения положительно определенных матриц
## Содержание
- [MKL](#Intel_Math_Kernel_Library (MKL))
- [OpenBLAS - LAPACK](#OpenBLAS_-_LAPACK)
- [Eigen](#Eigen)
- [Armadillo](#Armadillo)
- [NumPy](#NumPy)
- [CUDA cuBLAS и cuSOLVER](#CUDA_cuBLAS_и_cuSOLVER)
  
## Intel Math Kernel Library (MKL)
1. docker pull intel/oneapi-basekit
2. docker build -t intel/oneapi-basekit:latest -f Dockerfile.mkl . 
3. docker run --rm -ti -v mkl:/usr/share intel/oneapi-basekit:latest bash
4. ./mkl

ИЛИ ПРОЩЕ: docker run intel/oneapi-basekit
## OpenBLAS - LAPACK
1. docker build -t lapack:latest -f Dockerfile.lablas .
2. docker run lapack  200
## Eigen
Я все запихала в контейнер, но можно и без него.
Если без контейнера то так:
1. sudo apt update && sudo apt upgrade
2. sudo apt install libeigen3-dev
3. dpkg -L libeigen3-dev
4. Запускаем файл ei.cpp с помощью g++ -I /usr/include/eigen3/ ei.cpp -o ei  и  ./ei
Вот по этой ссылке делала я:  https://www.cyberithub.com/how-to-install-eigen3-on-ubuntu-20-04-lts-focal-fossa/

Если с контейнером, то так: папка eigen
1. docker build -t eigen:latest -f Dockerfile.Eigen .
2. docker run --rm -ti -v ei:/usr/share/myapp eigen:latest bash
3. ./ei

ИЛИ ПРОЩЕ:
docker run eigen 200

## Armadillo
Тут я тоже запихала в конейнер.
Без контейнера:
1. sudo apt update && sudo apt upgrade
2. sudo apt install cmake libopenblas-dev liblapack-dev
3. wget https://sourceforge.net/projects/arma/files/armadillo-14.0.1.tar.xz    (Скачать отсюда: https://arma.sourceforge.net/download.html    само Armadillo)     
4. Потом найти то, что скачали и разархивировать и перейти в эту разахивированную директорию:  cd arma*
5. cmake .
6. make
7. sudo make install
8. запускаем apm.cpp:  g++ arm.cpp -o arm -DARMA_DONT_USE_WRAPPER -lopenblas -llapack  и ./arm
Ссылка, по которой делала я: https://solarianprogrammer.com/2017/03/24/getting-started-armadillo-cpp-linear-algebra-windows-mac-linux/

С контейнером:
1. docker build -t armadillo:latest -f Dockerfile.arm .
2. docker run --rm -ti -v arm:/usr/share/app armadillo:latest bash
3. ./arm

ИЛИ ПРОЩЕ:
docker run armadillo 200
## NumPy
Без контейнера:
1. Запускаем файл NumPy.py:  python3 NumPy.py

С конейнером:
1. docker build -t num:latest -f Dockerfile.numpy .
2. docker run --rm -ti -v num:/usr/share/num num:latest bash
3. python3 NumPy.py

ИЛИ ПРОЩЕ:
docker run num 200
## CUDA cuBLAS и cuSOLVER
В процессе.....будет скоро...может быть

