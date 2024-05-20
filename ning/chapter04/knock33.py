file_path = 'neko.txt.mecab'
output_file_path = 'noun_phrases.txt'
noun_phrases = []

with open(file_path, 'r', encoding='utf-8') as file:
    sentence = []
    for line in file:
        line = line.strip()
        if line == 'EOS':
            if sentence:
                for i in range(1, len(sentence) - 1):
                    if (sentence[i-1]['pos'] == '名詞' and
                        sentence[i]['surface'] == 'の' and
                        sentence[i+1]['pos'] == '名詞'):
                        noun_phrase = sentence[i-1]['surface'] + sentence[i]['surface'] + sentence[i+1]['surface']
                        noun_phrases.append(noun_phrase)
                sentence = []
            continue

        if '\t' not in line:
            continue

        #形態素情報の抽出
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
with open(output_file_path, 'w', encoding='utf-8') as out_file:
    for phrase in noun_phrases:
        out_file.write(phrase + '\n')

