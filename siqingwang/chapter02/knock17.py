# 17. Distinct strings in the first column
distinct_values = set(df.iloc[:, 0])
print(distinct_values)

#!/bin/bash
# cut popular-names.txt | cut -f1 | sort | uniq