# Результат пилота задания 8

Дата и машина: 2026-10-30T09:00:16Z 
Intel(R) Xeon(R) Gold 6334 CPU @ 3.60GHz; logical_cpus = 16; governor = powersave; Ubuntu 22.04.5 LTS; kernel 6.8.0-106-generic

Commit кода: 00945a571619f21c36a80205359fec607a86d682

Файлы CSV и manifest: results/pilot8-r2-default.csv  и results/pilot8-r2-default.manifest.json results/pilot8-r2-single.csv и results/pilot8-r2-single.manifest.json

Governor и условия нагрузки: governor = powersave; измерения выполнялись последовательно одно за другим, без другой значимой нагрузки на сервер

Проверка корректности: 12 из 12 образов прошли; файл вывода: results/pilot8-validate256.log.

Unit-тесты: Ran 73 tests in 6.904s OK.

Результаты анализа:results/pilot8-r2-default.summary.csv и results/pilot8-r2-single.summary.csv (по 12 строк, count=3);
results/pilot8-r2-default.fits.csv и results/pilot8-r2-single.fits.csv(по 12 строк, все insufficient_sizes); 8 PNG на каждый блок в analysis/pilot8-r2-*.

## Время и режимы потоков

Все времена — медианы трёх повторов при n=2500, в секундах. Отношение равно single/default; больше 1 — стандартный режим быстрее.

Таблица медиан (n=2500, 3 повтора)

| Операция | Реализация | default, с | single, с | single/default |
|---|---|---|---|---|
| cholesky | mkl | 0.067924 | 0.254405 | 3.745 |
| cholesky | numpy | 0.186847 | 0.756900 | 4.051 |
| cholesky | openblas | 0.141624 | 0.374395 | 2.644 |
| lu | mkl | 0.105538 | 0.490282 | 4.646 |
| lu | numpy | 0.190611 | 0.612336 | 3.212 |
| lu | openblas | 0.340741 | 0.750857 | 2.204 |
| multiplication | mkl | 0.052337 | 0.339586 | 6.488 |
| multiplication | numpy | 0.068957 | 0.329791 | 4.783 |
| multiplication | openblas | 0.053466 | 0.313986 | 5.873 |
| svd | mkl | 1.474272 | 6.490544 | 4.403 |
| svd | numpy | 2.776321 | 6.529918 | 2.352 |
| svd | openblas | 2.615233 | 6.861452 | 2.624 |


Фактические пулы потоков в default: mkl- mkl/libmkl_rt:8, 
openblass-openblas/libopenblas:16, numpy-openblas/libopenblas:16;openblas/libopenblas:16; подтверждение :1 в single: mkl-mkl/libmkl_rt:1,openblass-openblas/libopenblas:1, numpy-openblas/libopenblas:1;openblas/libopenblas:1.

Максимальный RSS (значение, единицы и сочетание): 649880 КБ у mkl/svd/n=2500 режим default; 615988 КБ у mkl/svd/n=2500 режим single

Был ли swap или посторонняя нагрузка: нет (single swap=true: 0, default swap=true: 0)

Есть ли заметный разброс повторов: в single Q3/Q1 < 1.5 во всех группах. В default: lu/openblas при n=2500 имеет Q3/Q1 ≈ 4.83

Проверка совпадения commit/image ID и состава обоих блоков: commit 00945a571619f21c36a80205359fec607a86d682: один и тот же в двух manifest и во всех 72 строках CSV, image ID: все 12 совпадают, 36 строк в каждом блоке, 12 сочетаний × 3 повтора, n=2500 во всех строках

## Наблюдения и предварительный бюджет

1. Что отличается у SVD и multiplication при смене режима: отношение single/default больше у multiplication. Для multiplication и svd все три реализации сильно выигрывают в многопоточном режиме, MKL и OpenBLAS очень близкие значения отношения single/default  Однако отношение single/default для svd более неоднородное, openblas и numpy занижены по сравнению с mkl.   
2. Условный бюджет по кубической модели для предварительной сетки (время и единицы):

| n | r | T(n),с | r·T(n),с |
|---|---|---|---|
| 2500 | 10 | 8.1 | 80.7 |
| 5000 | 10 | 64.6 | 645.9 |
| 10000 | 10 | 516.7 | 5167.3 |
| 15000 | 3 | 1744.0 | 5231.9 |
| 20000 | 3 | 4133.8 | 12401.5 |

Итого чистых вычислений: 23527.3 с = 6.54 часов без сборки, генерации матриц, запуска контейнеров и тд

3. Ограничения оценки и что нужно уточнить до полного запуска: 
1)разное число потоков в default. mkl выбрал 8 потоков, а openblass - 16. В single число потоков зафиксировано и равно 1. 
2)в скриптах run_grid.py и test_run_grid.py была изменена проверка совпадений контрольных сумм на math.isclose с rel_tol=1e-12, abs_tol=1e-6, так как первая попытка упала на старой проверке контрольных сумм
3) В default lu/openblas при n=2500 имеет Q3/Q1 ≈ 4.83, что слишком много.


Если блок не завершился, вместо недостающих значений оставить пустые поля и указать файл журнала и причину. По неполному блоку бюджет не считать. Ожидаемый научный вывод заранее не задан.

