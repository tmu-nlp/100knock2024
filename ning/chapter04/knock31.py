file_path = 'neko.txt.mecab'
output_file_path = 'verbs.txt'
verbs = []  #動詞の表層形を格納するためのリスト

with open(file_path, 'r', encoding='utf-8') as file:
    sentence = []  
    for line in file:
        line = line.strip()
        if line == 'EOS':
            if sentence:
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
        
        #動詞の表層形の抽出
        if morph['pos'] == '動詞':
            verbs.append(morph['surface'])

#見やすくためのファイルに出力
with open(output_file_path, 'w', encoding='utf-8') as out_file:
    for verb in verbs:
        out_file.write(verb + '\n')


