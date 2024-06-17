#64. アナロジーデータでの実験Permalink
#単語アナロジーの評価データをダウンロードし，
# vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
# そのベクトルと類似度が最も高い単語と，その類似度を求めよ．
# 求めた単語と類似度は，各事例の末尾に追記せよ．

from tqdm import tqdm
file2 = './questions-words.txt'
output = './questions-words.txt'

# tqdm用のtotal数を先に調べておく
total = 0
with open(file2, 'r', encoding='utf-8') as f:
    for row in f:
        total += 1
    
category = ''
with open(file2, 'r', encoding='utf-8') as f1, \
        open(output, 'w', encoding='utf-8') as f2:
    for row in tqdm(f1, total=total):
        if row.startswith(':'):
            category = row.rstrip()[2:]
            continue
        else:
            cols = row.rstrip().split()
            word, similarity = model.most_similar(positive=[cols[1], cols[2]], negative=[cols[0]], topn=1)[0]
            f2.write('{}\t{}\t{}\t{}\n'.format(category, row.rstrip(), word, similarity))

"""
capital-common-countries	Athens Greece Baghdad Iraq	Iraqi	0.635187029838562
capital-common-countries	Athens Greece Bangkok Thailand	Thailand	0.7137669324874878
capital-common-countries	Athens Greece Beijing China	China	0.7235778570175171
capital-common-countries	Athens Greece Berlin Germany	Germany	0.6734622716903687
capital-common-countries	Athens Greece Bern Switzerland	Switzerland	0.4919748306274414
capital-common-countries	Athens Greece Cairo Egypt	Egypt	0.7527808547019958
capital-common-countries	Athens Greece Canberra Australia	Australia	0.583732545375824
capital-common-countries	Athens Greece Hanoi Vietnam	Viet_Nam	0.6276341676712036
capital-common-countries	Athens Greece Havana Cuba	Cuba	0.6460990905761719
capital-common-countries	Athens Greece Helsinki Finland	Finland	0.68999844789505
"""