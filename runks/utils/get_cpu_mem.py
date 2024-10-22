import re
import sys
from pathlib import Path

def analyze_log_file(file_path):
    max_cpu = 0
    max_mem = 0
    
    # Read the file and process each line
    with open(file_path, 'r') as f:
        for line in f:
            # Extract CPU percentage
            cpu_match = re.search(r'CPU: ([\d.]+)%', line)
            if cpu_match:
                cpu_value = float(cpu_match.group(1))
                max_cpu = max(max_cpu, cpu_value)
            
            # Extract Memory usage in GiB
            mem_match = re.search(r'Memory: ([\d.]+)GiB', line)
            if mem_match:
                mem_value = float(mem_match.group(1))
                max_mem = max(max_mem, mem_value)
    
    return max_cpu, max_mem

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <log_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).is_file():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    max_cpu, max_mem = analyze_log_file(file_path)
    print(f"{Path(file_path).name} {max_cpu:.2f} % {max_mem:.3f} GB")

if __name__ == "__main__":
    main()
