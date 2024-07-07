import MeCab

def morphology_map(file_parsed):
    with open(file_parsed) as mecab_parsed:
        mecab_parsed = mecab_parsed.read()
    mecab_parsed = mecab_parsed.lstrip('\n')
    lines = mecab_parsed.splitlines()
    res = []

    for line in lines:
        line_current = line.replace('\t',',').split(',')
        if line_current[0] == 'EOS':
            break
        else:
            dict = {
                    'surface' :line_current[0],
                    'base'    :line_current[-3],
                    'pos'     :line_current[1],
                    'pos1'    :line_current[2]
                    }
            res.append(dict)
    return res

if __name__ == "__main__":
    file = './neko.txt'
    file_parsed = './neko.txt.mecab'

    with open(file) as text, open(file_parsed, 'w') as text_parsed:
        mecab_tagger = MeCab.Tagger()
        mecab_parsed = mecab_tagger.parse(text.read())
        text_parsed.write(mecab_parsed)

    res = morphology_map(file_parsed)
    for item in res:
        print(item)