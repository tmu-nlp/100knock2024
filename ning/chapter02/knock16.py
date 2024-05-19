import numpy as np

with open("popular-names.txt", "r") as f:
    val = int(input("Enter the number of groups: "))
    lines = f.readlines()
    count = len(lines)
    num_list = range(count)
    div_list = np.array_split(num_list, val)
    
    for i, div in enumerate(div_list, 1):
        with open(f'{str(i).zfill(3)}.txt', 'w') as file:
            for j in div:
                file.write(lines[j])

