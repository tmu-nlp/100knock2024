from knock60 import load_word_vectors

def main():
    """
    メイン関数
    """
    file_path = 'GoogleNews-vectors-negative300.bin.gz'
    word_vectors = load_word_vectors(file_path)
    
    word = 'United_States'
    
    try:
        similar_words = word_vectors.most_similar(word, topn=10)
        print(f"単語 '{word}' と類似度の高い上位10語:")
        for similar_word, similarity in similar_words:
            print(f"単語: {similar_word}, 類似度: {similarity}")
    except KeyError as e:
        print(f"単語 '{e.args[0]}' はモデルに存在しません。")

if __name__ == '__main__':
    main()

"""
単語 'United_States' と類似度の高い上位10語:

単語: Unites_States, 類似度: 0.7877248525619507
単語: Untied_States, 類似度: 0.7541370987892151
単語: United_Sates, 類似度: 0.7400724291801453
単語: U.S., 類似度: 0.7310773730278015
単語: theUnited_States, 類似度: 0.6404393911361694
単語: America, 類似度: 0.6178409457206726
単語: UnitedStates, 類似度: 0.6167312264442444
単語: Europe, 類似度: 0.6132988333702087
単語: countries, 類似度: 0.6044804453849792
単語: Canada, 類似度: 0.6019070148468018
"""