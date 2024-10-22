import re
import csv
import os

def extract_times_and_correctness(input_files, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Файл', 'Размер матрицы', 'Время', 'Корректность'])  # Заголовки столбцов
        
        for input_file in input_files:
            times = []
            correctness = []
            matrix_size = None
            
            with open(input_file, 'r') as file:
                for line in file:
                    time_match = re.search(r'Время, затраченное на обращение матрицы: (\d+\.\d+) секунд', line)
                    correctness_match = re.search(r'Корректность обращения матрицы: (True|False)', line)
                    size_match = re.search(r'Running docker run num (\d+)', line)
                    
                    if time_match:
                        times.append(time_match.group(1))
                    if correctness_match:
                        correctness.append(correctness_match.group(1))
                    if size_match:
                        matrix_size = size_match.group(1)
            
            # Проверка, что количество значений времени и корректности совпадает
            if len(times) != len(correctness):
                raise ValueError(f"Количество значений времени и корректности не совпадает в файле {input_file}")
            
            # Запись данных в CSV
            for time, correct in zip(times, correctness):
                writer.writerow([os.path.basename(input_file), matrix_size, time, correct])

# Список файлов для обработки
input_files = ['resultnumCho.txt', 'resnummul.txt', 'resultnumQR.txt']  # Замените на реальные имена файлов
output_file = 'output.csv'

extract_times_and_correctness(input_files, output_file)
print(f"Данные успешно сохранены в {output_file}")
