noun_sequences = []

with open("neko.txt.mecab", 'r', encoding='utf-8') as file:
    sentence = [] 
    for line in file:
        line = line.strip() 
        if line == 'EOS':
            if sentence:
                #名詞の連接を抽出
                current_sequence = []
                for morph in sentence:
                    if morph['pos'] == '名詞':
                        current_sequence.append(morph['surface'])
                    else:
                        if len(current_sequence) > 1:
                            noun_sequences.append(''.join(current_sequence))
                        current_sequence = []
                if len(current_sequence) > 1:
                    noun_sequences.append(''.join(current_sequence))
                sentence = []
            continue

        if '\t' not in line:
            continue

        surface, details = line.split('\t')
        details = details.split(',')
        morph = {
            'surface': surface,
            'base': details[6] if len(details) > 6 else '*',
            'pos': details[0],
            'pos1': details[1]
        }
        sentence.append(morph)

#ファイルに出力
with open("longest_noun_sequences.txt", 'w', encoding='utf-8') as out_file:
    for sequence in noun_sequences:
        out_file.write(sequence + '\n')

