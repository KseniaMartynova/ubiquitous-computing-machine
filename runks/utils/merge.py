import re
import sys
import argparse

# def parse_first_file(content):
#     result = {}
#     for line in content.strip().split('\n'):
#         try:
#             name, size, time = re.match(r'(.+): (\d+) ([\d.]+)', line).groups()
#             base_name = name  # Оставляем .txt в имени
#             result[base_name] = {'size': size, 'time': time}
#         except AttributeError:
#             print(f"Warning: Could not parse line: {line}")
#     return result

# def parse_first_file(content):
#     result = {}
#     for line in content.strip().split('\n'):
#         try:
#             name, time = re.match(r'(.+): Среднее время ([\d.]+) секунд', line).groups()
#             base_name = name  # Оставляем .txt в имени
#             result[base_name] = {'time': time}  # Убираем 'size' из-за отсутствия в строке
#         except AttributeError:
#             print(f"Warning: Could not parse line: {line}")
#     return result

def parse_first_file(content):
    result = {}
    for line in content.strip().split('\n'):
        try:
            # Извлекаем имя файла, размер, время
            match = re.match(r'(.+_size_(\d+))\.txt: Среднее время ([\d.]+) секунд', line)
            if match:
                full_name, size, time = match.groups()
                result[full_name + '.txt'] = {'size': size, 'time': time}
            else:
                print(f"Warning: Could not parse line: {line}")
        except AttributeError:
            print(f"Warning: Could not parse line: {line}")
    return result

def parse_second_file(content):
    result = {}
    for line in content.strip().split('\n'):
        try:
            parts = line.split()
            name = parts[0].replace('_stats.log', '')
            cpu_percent = parts[1]
            memory = ' '.join(parts[3:5])
            result[name] = {'cpu': cpu_percent, 'memory': memory}
        except IndexError:
            print(f"Warning: Could not parse line: {line}")
    return result

def merge_data(first_content, second_content):
    # print(f"Parsing first file...")
    first_data = parse_first_file(first_content)
    # print(f"Found {len(first_data)} entries in first file")
    
    # print(f"Parsing second file...")
    second_data = parse_second_file(second_content)
    # print(f"Found {len(second_data)} entries in second file")
    
    merged_results = []
    for name in first_data:
        if name in second_data:
            base_name = name.replace('.txt', '').rsplit('_size_', 1)[0]
            size = first_data[name]['size']
            time = first_data[name]['time']
            cpu = second_data[name]['cpu']
            memory = second_data[name]['memory']
            merged_results.append(f"{base_name} {size} {time} {cpu} {memory}")
    
    # print(f"\nSuccessfully merged {len(merged_results)} entries")
    return merged_results

def main():
    parser = argparse.ArgumentParser(description='Merge benchmark timing and resource usage data')
    parser.add_argument('time_file', help='File containing timing results')
    parser.add_argument('resource_file', help='File containing CPU and memory usage')
    
    args = parser.parse_args()

    try:
        with open(args.time_file, 'r') as f:
            time_content = f.read()
        
        with open(args.resource_file, 'r') as f:
            resource_content = f.read()
        
        results = merge_data(time_content, resource_content)
        if results:
            # print("\nMerged results:")
            for result in sorted(results):
                print(result)
        else:
            # print("No matching entries found to merge!")
            pass
            
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
