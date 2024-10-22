import sys
import re

def calculate_average(file_name):
    times = []
    matrix_size = None

    # Объединенное регулярное выражение для поиска времени обращения матрицы
    regex_time = re.compile(
        r'Время, затраченное на обращение матрицы (?:размерности (\d+)x\1|(?:\(разложение Холецкого\)|\(QR-разложение\))?): ([\d.]+) секунд|'
        r'Время, затраченное на обращение матрицы: ([\d.]+) секунд|'
        r'Time to invert (\d+)x\4 matrix: ([\d.]+) seconds|'
        r'Time for (?:LU|QR) decomposition and inversion: ([\d.]+) seconds'
    )
    
    # Открываем файл
    with open(file_name, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()

        # Ищем строки с временем обращения матрицы
        match_time = regex_time.search(line)

        if match_time:
            if match_time.group(1):  # Совпадение для русского текста с указанием размера
                matrix_size = match_time.group(1)
                times.append(float(match_time.group(2)))
            elif match_time.group(2):  # Время для строки с разложением Холецкого или QR
                times.append(float(match_time.group(2)))
            elif match_time.group(3):  # Время для строки без указания разложения или размера
                times.append(float(match_time.group(3)))
            elif match_time.group(4):  # Совпадение для английского текста
                matrix_size = match_time.group(4)
                times.append(float(match_time.group(5)))
            elif match_time.group(6):  # Время для LU/QR разложения
                times.append(float(match_time.group(6)))

    # Выводим результат
    if times:
        avg_time = sum(times) / len(times)
        print(f"{file_name}: Среднее время {avg_time:.5f} секунд")
    else:
        print(f"{file_name}: Время не найдено")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python script_avg.py <file_name>")
        sys.exit(1)

    file_name = sys.argv[1]
    calculate_average(file_name)

