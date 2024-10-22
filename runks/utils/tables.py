import pandas as pd
import sys
from tabulate import tabulate

def format_value(x):
    if pd.isna(x):
        return ''
    elif isinstance(x, (int, float)):
        return f'{x:.6f}'
    else:
        return str(x)

def process_data(file_path):
    # Чтение данных из файла без заголовков
    df = pd.read_csv(file_path, sep='\s+', header=None)

    # Преобразуем последний столбец в строковый тип для обработки
    df[4] = df[4].astype(str)

    # Удаление последнего элемента (GB) из каждой строки
    df[4] = df[4].str.replace('GB', '', regex=False)  # Убираем 'GB'
    
    # Преобразуем оставшийся текст в числа
    df[4] = pd.to_numeric(df[4], errors='coerce')  # Преобразуем в числовой тип, ошибки будут заменены на NaN
    
    # Удаляем последний столбец (теперь он не нужен)
    df = df.drop(columns=[5], axis=1, errors='ignore')  # Удаляем 6-й столбец (с индексом 5)

    # Определение имен столбцов
    columns = ['Name', 'size', 'time,sec', 'cpu,%', 'Mem,GB']
    
    # Назначение имен столбцов
    df.columns = columns

    # Преобразование столбца 'size' в числовой тип
    df['size'] = pd.to_numeric(df['size'])

    # Сортировка DataFrame по размеру
    df = df.sort_values('size')

    return df

def create_table_for_size(df, size):
    size_df = df[df['size'] == size]

    # Сортировка по столбцу 'Name'
    size_df = size_df.sort_values('Name')

    # Поворот таблицы, чтобы 'Name' шли по горизонтали
    table = size_df[['Name', 'time,sec', 'cpu,%', 'Mem,GB']].set_index('Name')

    # Форматирование значений
    formatted_table = table.applymap(format_value)  # Применение форматирования к каждому элементу

    return formatted_table

def print_table(table, size):
    print(f"\nSize: {size}")
    print(tabulate(table, headers='keys', tablefmt='grid'))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file.txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    try:
        df = process_data(input_file)
        
        # Получаем уникальные размеры
        sizes = sorted(df['size'].unique())
        
        for size in sizes:
            table = create_table_for_size(df, size)
            print_table(table, size)

            # Сохранение результата в CSV файл для каждого размера
            output_file = f"{input_file.rsplit('.', 1)[0]}_{size}_output.csv"
            size_df = df[df['size'] == size]
            size_df.to_csv(output_file, index=False)
            print(f"\nResult for size {size} saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

