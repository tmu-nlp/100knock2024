from sklearn.feature_extraction.text import CountVectorizer
import joblib

df_train['TMP'] = 'train'
df_valid['TMP'] = 'valid'
df_test['TMP'] = 'test'

#データの結合
#concat([df1,df2])
#df[['a','b']] 中はインデックスのリスト
data = pd.concat([df_train[['TITLE','CATEGORY','TMP']],df_valid[['TITLE','CATEGORY','TMP']],df_test[['TITLE','CATEGORY','TMP']]],axis=0)
#reset_indexはdataのインデックスを更新している drop=Trueで元のインデックスを落とし、inplace=Trueでオブジェクトを変更している
data.reset_index(drop=True,inplace=True)
"""
CountVectorizerとTfidfVectorizerは、テキストデータをベクトル化するための手法です。
どちらを選ぶかは、具体的なタスクやデータによりますが、以下のポイントを考慮して選択することが一般的です。

CountVectorizer:
単語の出現回数をカウントする手法です。
文書内での単語の頻度を考慮しますが、文書全体の重要性は考慮しません。
テキスト分類やクラスタリングなど、単語の出現頻度が重要なタスクに適しています。
TfidfVectorizer:
TF-IDF (Term Frequency-Inverse Document Frequency) を計算する手法です。
単語の出現頻度と逆文書頻度を組み合わせて、単語の重要性を評価します。

"""
"""
(?u) は Unicode 文字列を扱うためのフラグで、正規表現内での文字列の解釈を Unicode モードに切り替えます。
\\b は単語の境界を表すメタキャラクタです。単語の先頭や末尾にマッチします。
\\w+ は一つ以上の単語文字（英数字やアンダースコア）にマッチします。例えば、apple や 123 などが該当します。
"""
vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
bag = vectorizer.fit_transform(data['TITLE'])

"""
bag の正体
CountVectorizerを使ってテキストデータをベクトル化した結果、
bagは疎行列 (sparse matrix) 形式になっています。
疎行列は非ゼロ要素が少ない行列で、メモリ効率のためにこの形式で保存されます。

疎行列を配列に変換する理由
疎行列 (sparse matrix) 形式は、scipy.sparse モジュールの csr_matrix（Compressed Sparse Row）などの形式で保存されています。
この形式はメモリ効率が良いですが、PandasのDataFrameには直接結合できません。
そのため、疎行列を通常の配列（dense array）に変換する必要があります。
"""
data = pd.concat([data, pd.DataFrame(bag.toarray())], axis=1)

"""
joblib.dumpは、Pythonのオブジェクトをファイルに保存するための関数
vectorizer.vocabulary_について
CountVectorizerのvocabulary_属性は、テキストデータ内の全単語と、それらの単語に割り当てられたインデックスの対応関係を保持する辞書です。
例えば、{'word1': 0, 'word2': 1, 'word3': 2, ...} のような形式です。
この情報を保存することで、将来の予測や解析時に同じ単語辞書を使うことができます。

保存の目的
再利用性: 一度作成した単語辞書を再利用できるため、新たに辞書を生成する手間が省けます。
一貫性の保持: モデルのトレーニングと予測で同じ辞書を使用することで、一貫性を保持できます。異なる辞書を使うと、単語のインデックスがずれて予測結果が正しくなくなる可能性があります。
効率性: 辞書の生成は計算コストがかかるため、一度生成した辞書を保存しておくことで効率的に処理が行えます。
"""
joblib.dump(vectorizer.vocabulary_, 'vocabulary_.joblib')


data_train = data.query('TMP=="train"').drop(['TITLE','TMP'], axis=1)
data_valid = data.query('TMP=="valid"').drop(['TITLE','TMP'], axis=1)
data_test = data.query('TMP=="test"').drop(['TITLE','TMP'], axis=1)

#CSV形式で書き出し
data_train.to_csv('data_train.feature.txt',sep='\t', index=False,header=False)
data_valid.to_csv('data_valid.feature.txt',sep='\t', index=False,header=False)
data_test.to_csv('data_test.feature.txt',sep='\t', index=False,header=False)
data_train.head()