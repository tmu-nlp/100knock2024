from knock41 import parse_chunks

def extract_noun_verb_dependency_pairs(sentences):
    """ 名詞を含む文節が動詞を含む文節に係るものを抽出する関数 """
    pairs = []
    for sentence in sentences:
        for chunk in sentence:
            if chunk.dst != -1:  # 係り先がある場合
                if any(morph.pos == '名詞' for morph in chunk.morphs) and any(morph.pos == '動詞' for morph in sentence[chunk.dst].morphs):
                    src_text = ''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号'])
                    dst_text = ''.join([morph.surface for morph in sentence[chunk.dst].morphs if morph.pos != '記号'])
                    if src_text and dst_text:  # 両方のテキストが空でない場合
                        pairs.append(f"{src_text}\t{dst_text}")
    return pairs

def main():
    file_path = 'ai.ja.txt.parsed'
    sentences = list(parse_chunks(file_path))
    noun_verb_pairs = extract_noun_verb_dependency_pairs(sentences)
    with open('knock43.txt', 'w', encoding='utf-8') as f:
        for pair in noun_verb_pairs:
            f.write(pair + '\n')

if __name__ == "__main__":
    main()
