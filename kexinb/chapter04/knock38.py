# task38. ヒストグラム
# 単語の出現頻度のヒストグラムを描け．ただし，横軸は出現頻度を表し，1から単語の出現頻度の最大値までの線形目盛とする．
# 縦軸はx軸で示される出現頻度となった単語の異なり数（種類数）である．

import matplotlib.pyplot as plt
import japanize_matplotlib


if __name__ == "__main__":
    with open('knock35_output.txt','r') as f:
        cnts = []
        for line in f:
            if line.strip():  # if not empty
                cnt = int(line.split(',')[0].strip('() '))
                cnts.append(cnt)
    
    plt.figure(figsize=(10, 6))
    plt.hist(cnts, bins=len(set(cnts)), color='blue', alpha=0.7)
    plt.title('Histogram of Word Counts')
    plt.xlabel('Word Counts (number of occurrences)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('knock38_output.png')
