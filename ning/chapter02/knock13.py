with open('col1.txt', 'r') as col1_file, open('col2.txt', 'r') as col2_file:
    col1_lines = col1_file.readlines()
    col2_lines = col2_file.readlines()

with open('combined12.txt', 'w') as combined_file:
    for col1_line, col2_line in zip(col1_lines, col2_lines):
        combined_file.write(f"{col1_line.strip()}\t{col2_line}")
