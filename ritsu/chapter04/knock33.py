from knock30 import parse_neko

def extract_noun_phrases(sentences):
    noun_phrases = []
    for sentence in sentences:
        for i in range(len(sentence) - 2):
            if sentence[i]['pos'] == '名詞' and sentence[i+1]['surface'] == 'の' and sentence[i+2]['pos'] == '名詞':
                noun_phrases.append(sentence[i]['surface'] + sentence[i+1]['surface'] + sentence[i+2]['surface'])
    return noun_phrases

if __name__ == "__main__":
    sentences = parse_neko('neko.txt.mecab')
    noun_phrases = extract_noun_phrases(sentences)
    print(noun_phrases[:10])  # 最初の10個の名詞句を表示