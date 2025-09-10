#!/bin/bash

mkdir -p result
cd result

# Список Python скриптов
scripts=(
    "numINV.py"
)

# Размеры матриц
sizes=(2500 5000 7500 10000 12500 15000 17500 20000)
# Количество запусков для каждого скрипта и размера
runs=10

# Функция для мониторинга ресурсов
monitor_resources() {
    pid=$1
    output_file=$2
    while ps -p $pid > /dev/null; do
        # Собираем данные: время, %CPU, RSS-память (KB), VSZ-память (KB)
        timestamp=$(date +%s)
        stats=$(ps -p $pid -o %cpu=,rss=,vsz=)
        echo "$timestamp $stats" >> "$output_file"
        sleep 1
    done
}

# Запуск Python скриптов
for script in "${scripts[@]}"; do
    for size in "${sizes[@]}"; do
        # Базовое имя для файлов результатов
        base_name="${script%.*}_size_${size}"
        
        for ((i=1; i<=runs; i++)); do
            echo "Запуск $script с размером $size, запуск $i..."
            
            # Файлы для результатов
            output_file="${base_name}.txt"
            stats_file="${base_name}_run_${i}_stats.log"
            
            # Запускаем Python-скрипт в фоне
            python3 "../$script" "$size" >> "$output_file" 2>&1 &
            pid=$!
            
            # Мониторим ресурсы
            monitor_resources $pid "$stats_file" &
            monitor_pid=$!
            
            # Ждем завершения скрипта
            wait $pid
            
            # Останавливаем мониторинг
            kill $monitor_pid 2>/dev/null
            
            echo "Завершено: $script, размер $size, запуск $i"
            echo "--------------------------------------" >> "$output_file"
        done
    done
done

cd ..
