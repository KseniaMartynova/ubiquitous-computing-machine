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
Библиотека Intel Math Kernel Library (MKL) — это набор высокопроизводительных математических функций, оптимизированных для работы на процессорах Intel. MKL предоставляет функции для линейной алгебры, быстрого преобразования Фурье, математических функций, статистики и других задач. Эта библиотека поддерживает многопоточность, что позволяет эффективно использовать многоядерные процессоры. Функции MKL автоматически распределяют вычисления между ядрами, что уменьшает время выполнения задач.Может быть использована в сочетании с другими библиотеками, такими как BLAS (Basic Linear Algebra Subprograms), LAPACK (Linear Algebra Package), FFTW (Fastest Fourier Transform in the West) и другими. Это обеспечивает высокую производительность и совместимость.
Итак, запустим образ из официального dockerhub intel.
'1. docker pull intel/oneapi-basekit'

Далее я создала dockerfile, в котором содержится все, что мне нужно для дальнейших вычилений, например, сама программа, которая производит обращение положительно определенной матрицы и замеряет время ее обращения.

2. docker build -t intel/oneapi-basekit:latest -f Dockerfile.mkl .

Для единичного запуска программы мы можем запустить контейнер и внутри уже ввести размер матрицы.

3. docker run --rm -ti -v mkl:/usr/share intel/oneapi-basekit:latest bash

4. ./mkl

Но на самом деле проще не заходить в контейнер и запустить следующее с нужным размером матрицы:

docker run intel/oneapi-basekit 200

Так как мне нужно было несколько раз подсчитать время обращения матрицы, был написан скрипт, который десять раз заапускает контейнер и записывает результаты в файл.
Вот запуск этого скрипта:

bash mkrun.sh | tee -a resmkl.txt

Результаты представлены в самом низу. 
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
8. запускаем arm.cpp:  g++ arm.cpp -o arm -DARMA_DONT_USE_WRAPPER -lopenblas -llapack  и ./arm
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
|1.|0.00963013|1.88359|0.0078014|0.0177394|0.010906|
|2.|0.00954745|1.84733|0.00808132|0.0183654|0.009947|
|3.|0.00944007|1.84766|0.00791611|0.0172747|0.010055|
|4.|0.00952895|1.84568|0.00781147|0.017478|0.009979|
|5.|0.0094517|1.84637|0.00769193|0.0171597|0.009962|
|6.|0.00955235|1.84986|0.00789403|0.0173177|0.010009|
|7.|0.00949818|1.85188|0.0078377|0.0176782|0.009890|
|8.|0.00940884|1.85076|0.00769071|0.0176869|0.010051|
|9.|0.00929534|1.84959|0.00769064|0.0176801|0.010117|
|10.|0.00934544|1.84579|0.00780599|0.0200063|0.010373|

Время обращения матрицы 1000 на 1000.
||Armadillo|Eigen|Lapack-OpenBlas|MKL|NumPY|
|--|---------|-----|---------------|---|-----|
|1.|0.0408857|14.1881|0.0488454|0.0314461|0.036915|
|2.|0.0400767|14.1684|0.027271|0.0285608|0.036681|
|3.|0.0400486|14.1694|0.0272823|0.0283974|0.036012|
|4.|0.040371|14.1685|0.0272818|0.0292272|0.0371224|
|5.|0.0399107|14.1716|0.0273705|0.028081|0.036774|
|6.|0.0403594|14.1691|0.0279127|0.0314063|0.041725|
|7.|0.040453|14.1695|0.0285193|0.0306483|0.036961|
|8.|0.0409319|14.1776|0.0273331|0.0285532|0.035921|
|9.|0.039873|14.1732|0.0343377|0.0308469|0.036551|
|10.|0.0401077|14.1736|0.027908|0.0311808|0.036249|

Время обращения матрицы 5000 на 5000.
||Armadillo|Eigen|Lapack-OpenBlas|MKL|NumPY|
|--|---------|-----|---------------|---|-----|
|1.|0.792006|неизвестно|0.675831|0.626723|1.025416|
|2.|0.788071|неизвестно|0.674122|0.602388|0.988560|
|3.|0.780386|неизвестно|0.676815|0.539019|0.995102|
|4.|0.785718|неизвестно|0.674146|0.803652|0.988205|
|5.|0.793708|неизвестно|0.673917|0.590867|1.014630|
|6.|0.800275|неизвестно|0.674592|0.542115|1.010876|
|7.|0.79115|неизвестно|0.671608|0.673194|1.041416|
|8.|0.784003|неизвестно|0.67442|0.552021|1.007140|
|9.|0.783363|неизвестно|0.674267|0.64314|1.011620|
|10.|0.789323|неизвестно|0.676193|0.538861|1.003036|

Время обращения матрицы 10 000 на 10 000.
||Armadillo|Eigen|Lapack-OpenBlas|MKL|NumPY|
|--|---------|-----|---------------|---|-----|
|1.|4.43055|неизвестно|4.13193|3.79721|6.976484|
|2.|4.4081|неизвестно|4.08334|3.92294|6.941000|
|3.|4.46715|неизвестно|4.16059|3.67781|6.944445|
|4.|4.46275|неизвестно|4.14197|3.84259|6.830090|
|5.|4.52823|неизвестно|4.13066|3.67603|6.856174|
|6.|4.45369|неизвестно|4.1148|3.71342|6.880401|
|7.|4.43212|неизвестно|4.16889|3.69516|6.900842|
|8.|4.43338|неизвестно|4.13052|3.86021|6.903779|
|9.|4.46111|неизвестно|4.14477|3.78372|7.027612|
|10.|4.39085|неизвестно|4.13681|3.89087|7.026535|

Время обращения матрицы 20 000 на 20 000.
||Armadillo|Eigen|Lapack-OpenBlas|MKL|NumPY|
|--|---------|-----|---------------|---|-----|
|1.|29.7498|неизвестно|27.6416|28.6198|52.407960|
|2.|30.2344|неизвестно|27.1703|28.0801|52.609518|
|3.|30.2236|неизвестно|27.8342|27.9994|52.327180|
|4.|30.0646|неизвестно|27.6613|27.953|52.362092|
|5.|30.6705|неизвестно|27.6842|28.3528|52.236765|
|6.|30.0366|неизвестно|27.5807|27.9798|52.376151|
|7.|30.4255|неизвестно|27.7714|28.1692|52.721389|
|8.|29.9864|неизвестно|27.7753|28.0445|52.727562|
|9.|30.4517|неизвестно|27.4314|27.7375|51.669924|
|10.|30.2366|неизвестно|27.7862|28.1186|52.824136|

Среднее:
|Размер Матрицы|Armadillo|Eigen|Lapack-OpenBlas|MKL|NumPY|
|--|---------|-----|---------------|---|-----|
|100×100|0,0008172634|0,01843804|0,0008078196|0,01232148|0,0017315|
|500×500|0,009469845|1,84579|0,00782213|0,01783864|0,0101289|
|1000×1000|0,04030177|14,1729|0,03040618|0,0298348|0,0370911|
|5000×5000|0,7888003|неизвестно|0,6745911|0,611198|1,0086001|
|10000×10000|4,446793|неизвестно|4,134428|3,785996|6,9287362|
|20000×20000|30,20797|неизвестно|27,63366|28,10547|52,4262677|
