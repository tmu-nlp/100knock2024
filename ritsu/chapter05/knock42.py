from knock41 import parse_chunks

def extract_dependency_pairs(sentences):
    """ 係り元と係り先の文節のテキストを抽出する関数 """
    pairs = []
    for sentence in sentences:
        for chunk in sentence:
            if chunk.dst != -1:  # 係り先がある場合
                src_text = ''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号'])
                dst_text = ''.join([morph.surface for morph in sentence[chunk.dst].morphs if morph.pos != '記号'])
                if src_text and dst_text:  # 両方のテキストが空でない場合
                    pairs.append(f"{src_text}\t{dst_text}")
    return pairs

def main():
    file_path = 'ai.ja.txt.parsed'
    sentences = list(parse_chunks(file_path))
    dependency_pairs = extract_dependency_pairs(sentences)
    with open('knock42.txt', 'w', encoding='utf-8') as f:
        for pair in dependency_pairs:
            f.write(pair + '\n')

if __name__ == "__main__":
    main()
