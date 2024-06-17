from knock60 import load_word_vectors

def main():
    """
    メイン関数
    """
    file_path = 'GoogleNews-vectors-negative300.bin.gz'
    word_vectors = load_word_vectors(file_path)
    
    word1 = 'Spain'
    word2 = 'Madrid'
    word3 = 'Athens' # Athensはギリシャの首都
    
    try:
        vector = word_vectors[word1] - word_vectors[word2] + word_vectors[word3]
        similar_words = word_vectors.similar_by_vector(vector, topn=10)
        
        print(f"単語 '{word1}' から '{word2}' を引き、'{word3}' を足したベクトルと類似度の高い上位10語:")
        for similar_word, similarity in similar_words:
            print(f"単語: {similar_word}, 類似度: {similarity}")
    except KeyError as e:
        print(f"単語 '{e.args[0]}' はモデルに存在しません。")

if __name__ == '__main__':
    main()

"""
単語 'Spain' から 'Madrid' を引き、'Athens' を足したベクトルと類似度の高い上位10語:

単語: Athens, 類似度: 0.7528456449508667
単語: Greece, 類似度: 0.6685471534729004
単語: Aristeidis_Grigoriadis, 類似度: 0.5495778322219849
単語: Ioannis_Drymonakos, 類似度: 0.5361456871032715
単語: Greeks, 類似度: 0.5351786613464355
単語: Ioannis_Christou, 類似度: 0.5330225825309753
単語: Hrysopiyi_Devetzi, 類似度: 0.5088489651679993
単語: Iraklion, 類似度: 0.5059264898300171
単語: Greek, 類似度: 0.5040615797042847
単語: Athens_Greece, 類似度: 0.5034109950065613
"""