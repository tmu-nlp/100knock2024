from gensim.models import KeyedVectors
import pickle

with open("drive/MyDrive/knock64.txt", "w") as o_file:
    with open("drive/MyDrive/questions-words.txt", "r") as t_file:
        for line in t_file:
            words = line.strip().split(" ")
            if len(words) != 4:
                continue
            word,value = model.most_similar(positive=[words[1], words[2]], negative=[words[0]])[0]
            o_file.write(f"{words[0]} {words[1]} {words[2]} {words[3]} {word} {value}\n")

# 8869までが意味的アナロジー
# 8870以降が文法的アナロジー
with open("drive/MyDrive/knock64.txt", "r") as f:
    with open("drive/MyDrive/knock64-semantic.txt", 'w') as o_file1:
        with open("drive/MyDrive/knock64-syntactic.txt", 'w') as o_file2:
            for i, line in enumerate(f):
                if i < 8869:
                    o_file1.write(line)
                else:
                    o_file2.write(line)