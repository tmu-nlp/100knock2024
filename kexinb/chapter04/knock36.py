# task36. 頻度上位10語
# 出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import matplotlib.pyplot as plt
import japanize_matplotlib

if __name__ == "__main__":
    with open('knock35_output.txt','r') as f:
        cnts = []
        chrs = []
        for i in range(10):
            line = f.readline()
            stripped = line.strip()[1:-2]
            cnt,chr = stripped.split(',')
            cnts.append(int(cnt.strip()))
            chrs.append(chr.strip().strip("'"))
    
    plt.figure(figsize=(10, 8))
    plt.bar(chrs, cnts, color='skyblue')
    plt.xlabel('Characters')  
    plt.ylabel('Frequency')  
    plt.title('Frequency of the Top 10 Characters')
    plt.savefig('knock36_output.png')
