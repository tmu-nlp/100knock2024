#64. アナロジーデータでの実験Permalink
#単語アナロジーの評価データをダウンロードし，
# vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
# そのベクトルと類似度が最も高い単語と，その類似度を求めよ．
# 求めた単語と類似度は，各事例の末尾に追記せよ．

#自然言語処理用のgensimパッケージ
from gensim.models import KeyedVectors
file = './GoogleNews-vectors-negative300.bin.gz'
#fileをword2vec形式で読み込み
model = KeyedVectors.load_word2vec_format(file,binary = True)

from tqdm import tqdm
file2 = './questions-words.txt'
output = './questions-words_similarity.txt'

# tqdm用のtotal数を先に調べておく
total = 0
with open(file2, 'r', encoding='utf-8') as f:
    for row in f:
        total += 1

category = ''
with open(file2, 'r', encoding='utf-8') as f1, \
        open(output, 'w', encoding='utf-8') as f2:
    for row in tqdm(f1, total=total):                                                                           #tqdmで進捗を表示
        if row.startswith(':'):                                                                                 #カテゴリーを表す行に関する処理
            category = row.rstrip()[2:]                                                                         #カテゴリーを取得
            continue
        else:                                                                                                   #各カテゴリー内での処理
            cols = row.rstrip().split()                                                                         #各行を空白区切りで行列に保存
            word, similarity = model.most_similar(positive=[cols[1], cols[2]], negative=[cols[0]], topn=1)[0]   #2,3行目を加算し、1行目を減算し、最も類似度の大きい単語とその類似度を保存
            f2.write('{}\t{}\t{}\t{}\n'.format(category, row.rstrip(), word, similarity))                       #ファイルへの書き込み
     

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