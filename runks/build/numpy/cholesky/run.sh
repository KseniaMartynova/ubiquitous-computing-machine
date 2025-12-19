#!/bin/bash

# Список контейнеров
containers=(
	"num_chol"
)

# Размеры матриц
sizes=(2500 5000 7500 10000 12500 15000 17500 20000)

# Количество запусков для каждого контейнера и размера
runs=10

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
