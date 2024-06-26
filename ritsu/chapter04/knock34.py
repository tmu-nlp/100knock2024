from knock30 import parse_neko

def extract_noun_sequences(sentences):
    noun_sequences = []
    for sentence in sentences:
        sequence = []
        for morph in sentence:
            if morph['pos'] == '名詞':
                sequence.append(morph['surface']) # 
            else:
                if len(sequence) > 1:
                    noun_sequences.append(''.join(sequence))
                sequence = []
        if len(sequence) > 1:
            noun_sequences.append(''.join(sequence))
    return noun_sequences

if __name__ == "__main__":
    sentences = parse_neko('neko.txt.mecab')
    noun_sequences = extract_noun_sequences(sentences)
    print(noun_sequences[:10])  # 最初の10個の名詞の連接を表示