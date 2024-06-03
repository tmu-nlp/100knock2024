# 18. Sort lines in descending order of the third column

df = pd.read_csv(file_path, delimiter='\t', header=None)

sorted_df = df.sort_values(by=df.columns[2], ascending=False)
print(sorted_df.head())