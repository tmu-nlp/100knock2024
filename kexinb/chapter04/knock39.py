# task39. Zipfの法則
# 単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．

import matplotlib.pyplot as plt
import japanize_matplotlib

if __name__ == "__main__":
    with open('knock35_output.txt','r') as f:
        cnts = []
        for line in f:
            if line.strip():  # if not empty
                number = int(line.split(',')[0].strip('() '))
                cnts.append(number)
    
        ranks = range(1, len(cnts) + 1)

    plt.loglog(ranks, cnts)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Zipf\'s Law')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('knock39_output.png')