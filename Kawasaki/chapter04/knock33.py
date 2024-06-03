import knock30

A_no_B =[]

for sentence in knock30.sentence_list:
    for i in range(len(sentence)-2): #if文で3つの要素を取るため-2をしている。
        if sentence[i]["pos"] == "名詞" and sentence[i+1]["surface"] == "の" and sentence[i+2]["pos"] == "名詞":
            no = sentence[i]["surface"] + sentence[i+1]["surface"] + sentence[i+2]["surface"] 
            A_no_B.append(no)

if __name__ == '__main__':
    print(A_no_B[:10])