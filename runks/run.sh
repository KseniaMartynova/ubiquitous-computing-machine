#!/bin/bash

mkdir result
cd result

# Список контейнеров
containers=(
       	#mkl_chol"
	#mkl_svd"
	#mkl_qr" 
	#mkl_lu"
	#mkl_mul"
	#lasvd_without_o"
	"lapack_svd"
	#lapack_chol"
	#lapack_gaus"
	#lapack_svd_wo_omp"
	#lapack_qr_wo_omp"
	#num_mul
	#num_qr
	#num_svd
	#num_inv
	#num_ch"
)

# Размеры матриц
sizes=(100 500 1000 5000 10000 20000)
# sizes=(100 500)

# Количество запусков для каждого контейнера и размера
runs=10

# Функция для мониторинга ресурсов
monitor_resources() {
    container_id=$1
    output_file=$2
    while [ "$(docker inspect -f '{{.State.Running}}' "$container_id")" == "true" ]; do
       #  Получаем статистику и записываем в файл
        docker stats --no-stream --format "{{.Name}}: CPU: {{.CPUPerc}}, Memory: {{.MemUsage}}" "$container_id" >> "${output_file}_stats.log"
        sleep 1
    done
}

# Запуск контейнеров
for container in "${containers[@]}"; do
    for size in "${sizes[@]}"; do
        # Создаем файл для вывода для текущего контейнера и размера
        output_file="${container}_size_${size}.txt"

        for ((i=1; i<=runs; i++)); do
            echo "Запуск контейнера $container с размером матрицы $size, запуск номер $i..."

            # Запускаем контейнер в фоновом режиме и получаем его ID
            container_id=$(docker run -d --rm "$container" "$size")
            #echo docker run -d --rm "$container" "$size"
	    

            # # Запускаем мониторинг в фоновом режиме
            monitor_resources "$container_id" "$output_file" & monitor_pid=$!

            # # Ожидаем завершения контейнера и записываем его вывод в файл
            docker logs -f "$container_id" >> "$output_file"

            # # Ждем завершения контейнера
            docker wait "$container_id"

            # # Останавливаем мониторинг
            kill $monitor_pid

            echo "Вывод контейнера $container с размером матрицы $size, запуск номер $i добавлен в $output_file"
        done
    done
done

cd ../
