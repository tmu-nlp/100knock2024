# 16. Split a file into N pieces

import pandas as pd
import numpy as np

file_path = '/Users/hoshikawakiyoru/Downloads/popular-names.txt'
df = pd.read_csv(file_path, delimiter='\t', header=None)

N = int(input('N = '))

pieces = np.array_split(df, N)

print(pieces)