#65. アナロジータスクでの正解率
# 64の実行結果を用い，
# 意味的アナロジー（semantic analogy）と
# 文法的アナロジー（syntactic analogy）の正解率を測定せよ．

"""
意味的アナロジー:単語の意味に基づいた類推問題
例「王 : 王女 = 男 : 女」

文法的アナロジー:単語の文法的な形式や変化に基づいた類推問題
例「走る : 走った = 食べる : 食べた」

意味的アナロジーの正解率 = 意味的アナロジーの正解数 / 意味的アナロジーの総数
文法的アナロジーの正解率 = 文法的アナロジーの正解数 / 文法的アナロジーの総数

"""


#自然言語処理用のgensimパッケージ
from gensim.models import KeyedVectors
file = '/GoogleNews-vectors-negative300.bin'

#fileをword2vec形式で読み込み
model = KeyedVectors.load_word2vec_format(file,binary = True)

output = '/questions-words.txt'

#カウントと正解数の初期化
sem_cnt = 0  # 意味的アナロジーのカウント数
sem_true= 0  # 意味的アナロジーの正解数
syn_cnt = 0  # 文法的アナロジーのカウント数
syn_true= 0  # 文法的アナロジーの正解数

#ファイルの読み込み
with open(output, 'r', encoding='utf-8') as f:
    for row in f:
        #行の前後の空白を取り除き、タブで分割してリスト化
        cols = row.strip().split('\t')
        # cols[0]: アナロジーの種類（意味的または文法的）
        # cols[1]: ターゲット
        # cols[2]: 予測結果

        #ターゲット単語を取得。ターゲットが複数単語の場合、最後の単語を取得
        target = cols[1].split()[-1]
        #予測単語を取得
        pred = cols[2]
        
        #意味的アナロジー:カテゴリが「gram」で始まらない場合
        if not cols[0].startswith('gram'):
            sem_cnt += 1
            #ターゲットと予測が一致する場合
            if target == pred:
                sem_true += 1
        #文法的アナロジー:カテゴリが「gram」で始まる場合
        else:
            syn_cnt += 1
            if target == pred:
                syn_true += 1

# 正解率の計算と表示
print('意味的アナロジーの正解率: {}'.format(sem_true / sem_cnt))
print('文法的アナロジーの正解率: {}'.format(syn_true / syn_cnt))

"""
出力結果
意味的アナロジーの正解率: 0.7308602999210734
文法的アナロジーの正解率: 0.7400468384074942
"""