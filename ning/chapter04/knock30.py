file_path = 'neko.txt.mecab'
output_path = 'neko_morpheme.txt'
sentences = [] 
sentence = []  

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip() 
        if line == 'EOS':
            if sentence:
                sentences.append(sentence)
                sentence = []
            continue

        if '\t' not in line:
            continue

        #形態素情報の抽出と格納
        surface, details = line.split('\t')
        details = details.split(',')
        morph = {
            'surface': surface,
            'base': details[6] if len(details) > 6 else '*',
            'pos': details[0],
            'pos1': details[1]
        }
        sentence.append(morph)

#出力
with open(output_path, 'w', encoding='utf-8') as out_file:
    for sentence in sentences:
        for morph in sentence:
            out_file.write(f"surface: {morph['surface']}, base: {morph['base']}, pos: {morph['pos']}, pos1: {morph['pos1']}\n")
        out_file.write("\n") 





