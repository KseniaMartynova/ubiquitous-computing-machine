# Сравнение производительности библиотек линейной алгебры для обращения положительно определенных матриц
## Содержание
- [MKL](#Intel_Math_Kernel_Library (MKL))
- [OpenBLAS - LAPACK](#OpenBLAS_-_LAPACK)
- [Eigen](#Eigen)
- [Armadillo](#Armadillo)
- [NumPy](#NumPy)
- [CUDA cuBLAS и cuSOLVER](#CUDA_cuBLAS_и_cuSOLVER)
- [Результаты](#Результаты)

(для запуска: bash run_lapack.sh | tee -a result_openBLAS_LAPACK_docker.log)
## Intel Math Kernel Library (MKL)
1. docker pull intel/oneapi-basekit
2. docker build -t intel/oneapi-basekit:latest -f Dockerfile.mkl . 
3. docker run --rm -ti -v mkl:/usr/share intel/oneapi-basekit:latest bash
4. ./mkl

ИЛИ ПРОЩЕ: docker run intel/oneapi-basekit

|заголовок|
|---------|
|тескт|

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
## Результаты

Время обращения для матрицы размером 100 на 100.
||Armadillo|Eigen|Lapack-OpenBlas|MKL|NumPY|
|--|---------|-----|---------------|---|-----|
|1.|0.000782807|0.018416|0.00082911|0.012094|0.001666|
|2.|0.000869859|0.0184076|0.000723671|0.0114652|0.001758|
|3.|0.000840406|0.0184943|0.000822186|0.0128776|0.001739|
|4.|0.000829538|0.0184049|0.000730876|0.0124448|0.002019|
|5.|0.000774271|0.0183887|0.000895428|0.0115591|0.001567|
|6.|0.000755921|0.0184364|0.000740555|0.0123872|0.001610|
|7.|0.000829167|0.0184487|0.000896712|0.0130546|0.001865|
|8.|0.000798261|0.0184081|0.000812954|0.0116407|0.001645|
|9.|0.000833738|0.0185227|0.000818725|0.0125338|0.001907|
|10.|0.000858666|0.018453|0.000807979|0.0131578|0.001539|


Время обращения матрицы размером 500 на 500.
||Armadillo|Eigen|Lapack-OpenBlas|MKL|NumPY|
|--|---------|-----|---------------|---|-----|
|1.|0.00963013|1.88359|0.0078014|0.0177394|
|2.|0.00954745|1.84733|0.00808132|0.0183654|
|3.|0.00944007|1.84766|0.00791611|0.0172747|
|4.|0.00952895|1.84568|0.00781147|0.017478|
|5.|0.0094517|1.84637|0.00769193|0.0171597|
|6.|0.00955235|1.84986|0.00789403|0.0173177|
|7.|0.00949818|1.85188|0.0078377|0.0176782|
|8.|0.00940884|1.85076|0.00769071|0.0176869|
|9.|0.00929534|1.84959|0.00769064|0.0176801|
|10.|0.00934544|1.84579|0.00780599|0.0200063|






