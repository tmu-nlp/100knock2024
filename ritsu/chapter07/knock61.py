from knock60 import load_word_vectors

def main():
    """
    メイン関数
    """
    file_path = 'GoogleNews-vectors-negative300.bin.gz'
    word_vectors = load_word_vectors(file_path)
    
    word1 = 'United_States'
    word2 = 'U.S.'
    
    try:
        cosine_similarity = word_vectors.similarity(word1, word2)
        print(f"単語 '{word1}' と '{word2}' のコサイン類似度: {cosine_similarity}")
    except KeyError as e:
        print(f"単語 '{e.args[0]}' はモデルに存在しません。")

if __name__ == '__main__':
    main()

"""
単語 'United_States' と 'U.S.' のコサイン類似度: 0.7310774326324463
"""